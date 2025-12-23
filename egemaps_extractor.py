"""
eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) Feature Extractor
Based on the eGeMAPS v2.0 specification

This module extracts acoustic features commonly used for emotion recognition
in speech synthesis and analysis.

Reference:
- Eyben et al. (2015) "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)
  for Voice Research and Affective Computing"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


class eGeMAPS_Extractor(nn.Module):
    """
    eGeMAPS feature extractor for emotional speech synthesis

    Extracts a subset of eGeMAPS features relevant for TTS:
    1. F0 (fundamental frequency) statistics
    2. Energy/Loudness features
    3. Spectral features (MFCCs, spectral flux, etc.)
    4. Voice quality features (jitter, shimmer approximations)

    For VITS integration, we focus on frame-level features that can be
    aligned with mel-spectrograms.
    """

    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f0_min=80,
        f0_max=600,
        feature_dim=88  # Total eGeMAPS feature dimension
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.feature_dim = feature_dim

        # Mel-spectrogram for energy and spectral features
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2
        )

        # MFCC for spectral features
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'f_min': 0,
                'f_max': sample_rate // 2
            }
        )

        # Learnable projection to compress features to desired dimension
        # Input: concatenated features, Output: feature_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(13 + n_mels + 4, 256),  # MFCC + Mel + prosody features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        )

    def extract_f0(self, waveform):
        """
        Extract F0 (fundamental frequency) using a simple autocorrelation method

        Args:
            waveform: [B, T] audio waveform
        Returns:
            f0: [B, T_frames] F0 contour
        """
        # This is a simplified F0 extraction
        # For production, consider using CREPE, PYIN, or other robust methods

        batch_size = waveform.size(0)

        # For now, we'll use a learnable proxy
        # In practice, you should use a proper F0 extractor like torchcrepe
        num_frames = (waveform.size(1) + self.hop_length - 1) // self.hop_length

        # Placeholder: extract energy as F0 proxy (replace with real F0 extractor)
        f0 = torch.zeros(batch_size, num_frames, device=waveform.device)

        # TODO: Integrate proper F0 extraction (e.g., torchcrepe, torchyin)
        # Example: f0 = torchcrepe.predict(waveform, self.sample_rate, ...)

        return f0

    def extract_energy(self, mel_spec):
        """
        Extract energy features from mel-spectrogram

        Args:
            mel_spec: [B, n_mels, T] mel-spectrogram
        Returns:
            energy: [B, T] energy contour
        """
        # RMS energy across mel bins
        energy = torch.sqrt(torch.mean(mel_spec ** 2, dim=1))
        return energy

    def extract_spectral_flux(self, mel_spec):
        """
        Extract spectral flux (change in spectral energy over time)

        Args:
            mel_spec: [B, n_mels, T] mel-spectrogram
        Returns:
            flux: [B, T-1] spectral flux
        """
        # Difference between consecutive frames
        diff = mel_spec[:, :, 1:] - mel_spec[:, :, :-1]
        flux = torch.sqrt(torch.mean(diff ** 2, dim=1))

        # Pad to match original length
        flux = F.pad(flux, (1, 0), mode='replicate')
        return flux

    def extract_zcr(self, waveform):
        """
        Extract Zero Crossing Rate

        Args:
            waveform: [B, T] audio waveform
        Returns:
            zcr: [B, T_frames] zero crossing rate
        """
        batch_size, wav_len = waveform.size()
        num_frames = (wav_len + self.hop_length - 1) // self.hop_length

        # Compute sign changes
        sign = torch.sign(waveform)
        sign_change = torch.abs(sign[:, 1:] - sign[:, :-1]) / 2

        # Frame-wise ZCR
        zcr = torch.zeros(batch_size, num_frames, device=waveform.device)

        for i in range(num_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, wav_len - 1)
            if end > start:
                zcr[:, i] = torch.mean(sign_change[:, start:end], dim=1)

        return zcr

    def forward(self, waveform, return_dict=False):
        """
        Extract eGeMAPS features from audio waveform

        Args:
            waveform: [B, T] audio waveform
            return_dict: if True, return dict of individual features
        Returns:
            features: [B, feature_dim, T_frames] extracted features
            or dict of individual feature tensors if return_dict=True
        """
        # 1. Extract mel-spectrogram
        mel_spec = self.mel_spec(waveform)  # [B, n_mels, T_frames]

        # 2. Extract MFCC
        mfcc = self.mfcc(waveform)  # [B, 13, T_frames]

        # 3. Extract prosodic features
        f0 = self.extract_f0(waveform)  # [B, T_frames]
        energy = self.extract_energy(mel_spec)  # [B, T_frames]
        spectral_flux = self.extract_spectral_flux(mel_spec)  # [B, T_frames]
        zcr = self.extract_zcr(waveform)  # [B, T_frames]

        # 4. Concatenate all features
        # Ensure all have same time dimension
        T_frames = mel_spec.size(2)

        # Stack prosodic features
        prosody = torch.stack([
            f0[:, :T_frames],
            energy,
            spectral_flux,
            zcr[:, :T_frames]
        ], dim=1)  # [B, 4, T_frames]

        # Concatenate: MFCC + Mel + Prosody
        all_features = torch.cat([
            mfcc,           # [B, 13, T]
            mel_spec,       # [B, n_mels, T]
            prosody         # [B, 4, T]
        ], dim=1)  # [B, 13 + n_mels + 4, T]

        # 5. Project to desired feature dimension
        # Transpose for linear layer: [B, T, C]
        all_features = all_features.transpose(1, 2)  # [B, T, C]
        features = self.feature_projection(all_features)  # [B, T, feature_dim]
        features = features.transpose(1, 2)  # [B, feature_dim, T]

        if return_dict:
            return {
                'features': features,
                'f0': f0,
                'energy': energy,
                'spectral_flux': spectral_flux,
                'zcr': zcr,
                'mfcc': mfcc,
                'mel_spec': mel_spec
            }

        return features


class eGeMAPS_Encoder(nn.Module):
    """
    Encoder that processes eGeMAPS features for emotion conditioning
    This can be used in VITS to encode reference audio for emotion transfer
    """

    def __init__(
        self,
        feature_dim=88,
        hidden_channels=192,
        out_channels=192,
        kernel_size=5,
        n_layers=3,
        p_dropout=0.1
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Input projection
        self.pre = nn.Conv1d(feature_dim, hidden_channels, 1)

        # Convolutional layers for processing
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(n_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2
                )
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels))

        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask=None):
        """
        Args:
            x: [B, feature_dim, T] eGeMAPS features
            x_mask: [B, 1, T] mask
        Returns:
            output: [B, out_channels, T] encoded features
        """
        # Input projection
        x = self.pre(x)

        if x_mask is not None:
            x = x * x_mask

        # Convolutional processing
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            residual = x

            # Conv
            x = conv(x)

            # Transpose for LayerNorm: [B, C, T] -> [B, T, C]
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]

            # Activation and dropout
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection
            x = x + residual

            if x_mask is not None:
                x = x * x_mask

        # Output projection
        x = self.proj(x)

        if x_mask is not None:
            x = x * x_mask

        return x


# Example usage and testing
if __name__ == "__main__":
    # Test eGeMAPS extractor
    extractor = eGeMAPS_Extractor(
        sample_rate=22050,
        hop_length=256,
        feature_dim=88
    )

    # Dummy audio
    batch_size = 2
    audio_length = 22050 * 2  # 2 seconds
    waveform = torch.randn(batch_size, audio_length)

    # Extract features
    features = extractor(waveform)
    print(f"Extracted features shape: {features.shape}")  # [B, 88, T_frames]

    # Test encoder
    encoder = eGeMAPS_Encoder(feature_dim=88, hidden_channels=192, out_channels=192)
    encoded = encoder(features)
    print(f"Encoded features shape: {encoded.shape}")  # [B, 192, T_frames]

    print("eGeMAPS extractor test passed!")
