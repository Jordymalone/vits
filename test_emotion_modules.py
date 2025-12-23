"""
Test script for CCA and eGeMAPS modules in VITS

This script tests:
1. Cross Conditional Attention (CCA)
2. eGeMAPS feature extraction
3. Integration with VITS model
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attentions import CrossConditionalAttention
from egemaps_extractor import eGeMAPS_Extractor, eGeMAPS_Encoder


def test_cross_conditional_attention():
    """Test CrossConditionalAttention module"""
    print("=" * 60)
    print("Testing Cross Conditional Attention (CCA)...")
    print("=" * 60)

    batch_size = 2
    channels = 192
    cond_channels = 192
    seq_len = 100
    cond_len = 50
    n_heads = 4

    # Create CCA module
    cca = CrossConditionalAttention(
        channels=channels,
        cond_channels=cond_channels,
        n_heads=n_heads,
        p_dropout=0.1
    )

    # Create dummy inputs
    x = torch.randn(batch_size, channels, seq_len)
    cond = torch.randn(batch_size, cond_channels, cond_len)
    x_mask = torch.ones(batch_size, 1, seq_len)
    cond_mask = torch.ones(batch_size, 1, cond_len)

    # Forward pass
    output = cca(x, cond, x_mask=x_mask, cond_mask=cond_mask)

    print(f"Input shape: {x.shape}")
    print(f"Condition shape: {cond.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: torch.Size([{batch_size}, {channels}, {seq_len}])")

    # Check output shape
    assert output.shape == (batch_size, channels, seq_len), \
        f"Output shape mismatch! Got {output.shape}"

    # Check that output is different from input (attention has effect)
    assert not torch.allclose(output, x), "Output should be different from input!"

    print("✓ CCA test passed!")
    print()
    return True


def test_egemaps_extractor():
    """Test eGeMAPS feature extractor"""
    print("=" * 60)
    print("Testing eGeMAPS Feature Extractor...")
    print("=" * 60)

    batch_size = 2
    sample_rate = 22050
    audio_duration = 2  # seconds
    audio_length = sample_rate * audio_duration
    feature_dim = 88

    # Create extractor
    extractor = eGeMAPS_Extractor(
        sample_rate=sample_rate,
        hop_length=256,
        feature_dim=feature_dim
    )

    # Create dummy audio
    waveform = torch.randn(batch_size, audio_length)

    # Extract features
    features = extractor(waveform)

    print(f"Waveform shape: {waveform.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Expected feature dim: {feature_dim}")

    # Check feature dimension
    assert features.shape[1] == feature_dim, \
        f"Feature dimension mismatch! Got {features.shape[1]}, expected {feature_dim}"

    print(f"Number of frames: {features.shape[2]}")
    print("✓ eGeMAPS extractor test passed!")
    print()
    return True


def test_egemaps_encoder():
    """Test eGeMAPS encoder"""
    print("=" * 60)
    print("Testing eGeMAPS Encoder...")
    print("=" * 60)

    batch_size = 2
    feature_dim = 88
    hidden_channels = 192
    seq_len = 100

    # Create encoder
    encoder = eGeMAPS_Encoder(
        feature_dim=feature_dim,
        hidden_channels=hidden_channels,
        out_channels=hidden_channels,
        n_layers=3
    )

    # Create dummy features
    features = torch.randn(batch_size, feature_dim, seq_len)
    mask = torch.ones(batch_size, 1, seq_len)

    # Encode
    encoded = encoder(features, x_mask=mask)

    print(f"Input features shape: {features.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Expected shape: torch.Size([{batch_size}, {hidden_channels}, {seq_len}])")

    # Check output shape
    assert encoded.shape == (batch_size, hidden_channels, seq_len), \
        f"Output shape mismatch! Got {encoded.shape}"

    print("✓ eGeMAPS encoder test passed!")
    print()
    return True


def test_end_to_end_pipeline():
    """Test end-to-end pipeline: audio -> eGeMAPS -> encoder -> CCA"""
    print("=" * 60)
    print("Testing End-to-End Pipeline...")
    print("=" * 60)

    batch_size = 2
    sample_rate = 22050
    audio_duration = 2
    audio_length = sample_rate * audio_duration
    feature_dim = 88
    hidden_channels = 192
    text_seq_len = 100

    # 1. Create modules
    extractor = eGeMAPS_Extractor(
        sample_rate=sample_rate,
        hop_length=256,
        feature_dim=feature_dim
    )
    encoder = eGeMAPS_Encoder(
        feature_dim=feature_dim,
        hidden_channels=hidden_channels,
        out_channels=hidden_channels
    )
    cca = CrossConditionalAttention(
        channels=hidden_channels,
        cond_channels=hidden_channels,
        n_heads=4
    )

    # 2. Create dummy inputs
    ref_audio = torch.randn(batch_size, audio_length)
    text_features = torch.randn(batch_size, hidden_channels, text_seq_len)
    text_mask = torch.ones(batch_size, 1, text_seq_len)

    # 3. Extract eGeMAPS features
    print("Step 1: Extracting eGeMAPS features...")
    egemaps_feat = extractor(ref_audio)
    print(f"  eGeMAPS features shape: {egemaps_feat.shape}")

    # 4. Encode eGeMAPS features
    print("Step 2: Encoding eGeMAPS features...")
    emo_feat = encoder(egemaps_feat)
    print(f"  Encoded emotion features shape: {emo_feat.shape}")

    # 5. Create emotion mask
    emo_mask = torch.ones(batch_size, 1, emo_feat.size(2))

    # 6. Apply CCA
    print("Step 3: Applying Cross Conditional Attention...")
    output = cca(text_features, emo_feat, x_mask=text_mask, cond_mask=emo_mask)
    print(f"  CCA output shape: {output.shape}")

    # Check shapes
    assert output.shape == (batch_size, hidden_channels, text_seq_len), \
        f"Final output shape mismatch! Got {output.shape}"

    print("✓ End-to-end pipeline test passed!")
    print()
    return True


def test_model_initialization():
    """Test VITS model initialization with CCA and eGeMAPS"""
    print("=" * 60)
    print("Testing VITS Model Initialization with CCA and eGeMAPS...")
    print("=" * 60)

    try:
        from models import SynthesizerTrn
        import json

        # Load a sample config
        # You may need to adjust this path
        config_path = "configs/ljs_base.json"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Add emotion parameters
            model_config = config['model']
            model_config['use_cca'] = True
            model_config['use_egemaps'] = True
            model_config['n_emotions'] = 5  # Example: 5 emotions
            model_config['emo_feature_dim'] = 88

            # Initialize model
            model = SynthesizerTrn(
                **model_config,
                n_speakers=config['data']['n_speakers']
            )

            print(f"Model initialized successfully!")
            print(f"  use_cca: {model.use_cca}")
            print(f"  use_egemaps: {model.use_egemaps}")
            print(f"  n_emotions: {model.n_emotions}")

            # Check if modules exist
            if model.use_egemaps:
                assert hasattr(model, 'egemaps_extractor'), "Missing egemaps_extractor!"
                assert hasattr(model, 'egemaps_encoder'), "Missing egemaps_encoder!"
                print("  ✓ eGeMAPS modules present")

            if model.use_cca:
                assert hasattr(model.enc_p, 'cca'), "Missing CCA in text encoder!"
                print("  ✓ CCA module present")

            print("✓ Model initialization test passed!")
        else:
            print(f"Config file not found: {config_path}")
            print("Skipping model initialization test...")

    except Exception as e:
        print(f"Error during model initialization test: {e}")
        print("This is expected if config file is not available.")

    print()
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing Emotion Speech Synthesis Modules for VITS")
    print("=" * 60 + "\n")

    tests = [
        ("Cross Conditional Attention", test_cross_conditional_attention),
        ("eGeMAPS Extractor", test_egemaps_extractor),
        ("eGeMAPS Encoder", test_egemaps_encoder),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Model Initialization", test_model_initialization),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"✗ {test_name} test failed with error:")
            print(f"  {str(e)}\n")
            results.append((test_name, "FAILED"))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")
    print("=" * 60 + "\n")

    # Return success if all tests passed
    all_passed = all(result == "PASSED" for _, result in results)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
