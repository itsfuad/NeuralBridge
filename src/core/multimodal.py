from typing import Dict, Any
from .interfaces import MultiModalInterface
import base64
import io
from PIL import Image
import numpy as np
import torch
import torchaudio
import transformers

class MultiModalProcessor(MultiModalInterface):
    def __init__(self):
        self.image_processor = transformers.AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.text_processor = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.audio_processor = transformers.AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def process_text(self, text: str) -> str:
        """Process text input."""
        # Tokenize and process text
        tokens = self.text_processor(text, return_tensors="pt")
        return tokens

    def process_image(self, image_data: bytes) -> str:
        """Process image input."""
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Process image
        inputs = self.image_processor(image, return_tensors="pt")
        return inputs

    def process_audio(self, audio_data: bytes) -> str:
        """Process audio input."""
        # Convert bytes to audio tensor
        audio_tensor = self._bytes_to_audio_tensor(audio_data)
        
        # Process audio
        inputs = self.audio_processor(audio_tensor, sampling_rate=16000, return_tensors="pt")
        return inputs

    def _bytes_to_audio_tensor(self, audio_data: bytes) -> torch.Tensor:
        """Convert audio bytes to tensor."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Reshape to match expected format (batch_size, sequence_length)
        audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample to 16kHz if needed
        if audio_tensor.shape[1] != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio_tensor.shape[1],
                new_freq=16000
            )
            audio_tensor = resampler(audio_tensor)
        
        return audio_tensor

class MultiModalManager:
    def __init__(self):
        self.processor = MultiModalProcessor()

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal input."""
        result = {}
        
        if "text" in input_data:
            result["text"] = self.processor.process_text(input_data["text"])
            
        if "image" in input_data:
            image_data = base64.b64decode(input_data["image"])
            result["image"] = self.processor.process_image(image_data)
            
        if "audio" in input_data:
            audio_data = base64.b64decode(input_data["audio"])
            result["audio"] = self.processor.process_audio(audio_data)
            
        return result

    def combine_modalities(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine different modalities into a single representation."""
        if not processed_data:
            return processed_data

        # Initialize the combined representation
        combined = {
            "embeddings": [],
            "attention_masks": [],
            "modality_types": []
        }

        # Process each modality
        if "text" in processed_data:
            text_data = processed_data["text"]
            combined["embeddings"].append(text_data["input_ids"])
            combined["attention_masks"].append(text_data["attention_mask"])
            combined["modality_types"].append("text")

        if "image" in processed_data:
            image_data = processed_data["image"]
            combined["embeddings"].append(image_data["pixel_values"])
            combined["attention_masks"].append(torch.ones_like(image_data["pixel_values"][:, 0]))
            combined["modality_types"].append("image")

        if "audio" in processed_data:
            audio_data = processed_data["audio"]
            combined["embeddings"].append(audio_data["input_values"])
            combined["attention_masks"].append(audio_data["attention_mask"])
            combined["modality_types"].append("audio")

        # Pad sequences to the same length
        max_length = max(emb.shape[1] for emb in combined["embeddings"])
        padded_embeddings = []
        padded_masks = []

        for emb, mask in zip(combined["embeddings"], combined["attention_masks"]):
            if emb.shape[1] < max_length:
                # Pad embedding
                padding = torch.zeros((emb.shape[0], max_length - emb.shape[1], emb.shape[2]))
                padded_emb = torch.cat([emb, padding], dim=1)
                padded_embeddings.append(padded_emb)

                # Pad attention mask
                padding = torch.zeros((mask.shape[0], max_length - mask.shape[1]))
                padded_mask = torch.cat([mask, padding], dim=1)
                padded_masks.append(padded_mask)
            else:
                padded_embeddings.append(emb)
                padded_masks.append(mask)

        # Stack all modalities
        combined["embeddings"] = torch.stack(padded_embeddings, dim=1)
        combined["attention_masks"] = torch.stack(padded_masks, dim=1)

        return combined 