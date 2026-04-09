"""
Ensemble Module Unit Tests.

This module is used to test various functionalities of the Ensemble module, verifying code correctness.

Test Content:
    1. test_name_parser: Test TSGymNameParser name parsing and script name generation
    2. test_find_script: Test script retrieval functionality
    3. test_script_parser: Test ScriptParser script parsing and parameter merging
    4. test_check_existing_results: Test existing results checking functionality

Test Data:
    Uses 5 TSGym model names as test samples, covering different configuration combinations.

Usage:
    python meta/evaluation/test_ensemble.py

Author: TSGym
"""
"""
Test Ensemble Module

Uses 5 scripts from /data/nishome/user1/chaochuan/TSGym_benchmark/scripts/long_term_forecast/ETTh1_script/gym_MLP for testing
"""

import os
import sys

# Add project path
sys.path.insert(0, '/data/nishome/user1/chaochuan/TSGym_benchmark')

from meta.evaluation.ensemble import (
    TSGymNameParser,
    ScriptParser,
    EnsembleRunner,
    run_ensemble
)


# ==================== Test Data ====================
# TSGym names generated from 5 scripts
TEST_TOPK_NAMES = [
    "LTF_TSGym1000000_False_False_DishTS_MoEMA_False_series-encoding_MLP_DNN_null_True_False_True_ETTh1_ftM_sl192_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfFreDFLoss_lr0.0001_lrscosine_0",
    "LTF_TSGym1000001_False_False_RevIN_DFT_True_series-encoding_MLP_NormLin_null_True_False_True_ETTh1_ftM_sl192_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfMSE_lr0.0001_lrscosine_0",
    "LTF_TSGym1000002_True_False_RevIN_MoEMA_False_series-encoding_MLP_NormLin_sparse-attention_True_False_True_ETTh1_ftM_sl96_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfMAE_lr0.0001_lrscosine_0",
    "LTF_TSGym1000003_True_True_DishTS_MA_False_series-encoding_MLP_NormLin_sparse-attention_True_False_False_ETTh1_ftM_sl192_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfMAE_lr0.0001_lrscosine_0",
    "LTF_TSGym1000004_True_True_None_DFT_True_series-encoding_MLP_NormLin_null_True_False_False_ETTh1_ftM_sl512_ll48_pl96_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfFreDFLoss_lr0.0001_lrscosine_0",
]

# Corresponding script names (for verification)
EXPECTED_SCRIPT_NAMES = [
    "TSGym1000000_False_False_DishTS_MoEMA_False_series-encoding_MLP_DNN_null_True_False_True_HP_192_64-256_2_30_FreDFLoss_0.0001_cosine_96.sh",
    "TSGym1000001_False_False_RevIN_DFT_True_series-encoding_MLP_NormLin_null_True_False_True_HP_192_64-256_2_30_MSE_0.0001_cosine_96.sh",
    "TSGym1000002_True_False_RevIN_MoEMA_False_series-encoding_MLP_NormLin_sparse-attention_True_False_True_HP_96_64-256_2_30_MAE_0.0001_cosine_96.sh",
    "TSGym1000003_True_True_DishTS_MA_False_series-encoding_MLP_NormLin_sparse-attention_True_False_False_HP_192_64-256_2_30_MAE_0.0001_cosine_96.sh",
    "TSGym1000004_True_True_None_DFT_True_series-encoding_MLP_NormLin_null_True_False_False_HP_512_64-256_2_30_FreDFLoss_0.0001_cosine_96.sh",
]

SCRIPTS_ROOT = "/data/nishome/user1/chaochuan/TSGym_benchmark/scripts/long_term_forecast/ETTh1_script/gym_MLP"
PREDLEN = 96


def test_name_parser():
    """Test TSGymNameParser"""
    print("=" * 60)
    print("Testing TSGymNameParser")
    print("=" * 60)

    all_passed = True

    for i, name in enumerate(TEST_TOPK_NAMES):
        print(f"\n--- Test {i+1}: {name[:50]}... ---")

        parsed = TSGymNameParser.parse(name)
        script_name = TSGymNameParser.build_script_name(parsed, PREDLEN)
        expected = EXPECTED_SCRIPT_NAMES[i]

        print(f"  TSGym ID: {parsed.tsgym_id}")
        print(f"  Dataset: {parsed.dataset}")
        print(f"  Seq Len: {parsed.seq_len}")
        print(f"  Pred Len: {parsed.pred_len}")
        print(f"  D Model: {parsed.d_model}")
        print(f"  D FF: {parsed.d_ff}")
        print(f"  E Layers: {parsed.e_layers}")
        print(f"  Loss: {parsed.loss}")
        print(f"  Generated Script Name: {script_name}")
        print(f"  Expected Script Name: {expected}")

        if script_name == expected:
            print(f"  ✓ Passed")
        else:
            print(f"  ✗ Failed")
            all_passed = False

    return all_passed


def test_find_script():
    """Test script retrieval"""
    print("\n" + "=" * 60)
    print("Testing Script Retrieval")
    print("=" * 60)

    runner = EnsembleRunner(SCRIPTS_ROOT)
    all_passed = True

    for i, name in enumerate(TEST_TOPK_NAMES):
        script_path = runner.find_script(name, PREDLEN)

        if script_path and os.path.exists(script_path):
            print(f"  ✓ Found script: {os.path.basename(script_path)}")
        else:
            print(f"  ✗ Script not found: {EXPECTED_SCRIPT_NAMES[i]}")
            all_passed = False

    return all_passed


def test_script_parser():
    """Test script parser"""
    print("\n" + "=" * 60)
    print("Testing ScriptParser")
    print("=" * 60)

    runner = EnsembleRunner(SCRIPTS_ROOT)
    all_passed = True

    for name in TEST_TOPK_NAMES[:2]:  # Only test first 2
        print(f"\n--- Test: {name[:50]}... ---")

        script_path = runner.find_script(name, PREDLEN)
        if not script_path:
            print(f"  ✗ Script does not exist")
            all_passed = False
            continue

        script_params = ScriptParser.parse_script(script_path)
        print(f"  Parsed parameters:")
        print(f"    root_path: {script_params.get('root_path')}")
        print(f"    data_path: {script_params.get('data_path')}")
        print(f"    model_id: {script_params.get('model_id')}")
        print(f"    enc_in: {script_params.get('enc_in')}")
        print(f"    dec_in: {script_params.get('dec_in')}")
        print(f"    c_out: {script_params.get('c_out')}")

        # Merge parameters
        parsed = TSGymNameParser.parse(name)
        args = ScriptParser.merge_args(parsed, script_params, PREDLEN)

        print(f"  Merged args:")
        print(f"    model: {args.model}")
        print(f"    data: {args.data}")
        print(f"    pred_len: {args.pred_len}")
        print(f"    ensemble_mode: {args.ensemble_mode}")
        print(f"    ensemble_save_dir: {args.ensemble_save_dir}")

        if args.ensemble_mode and args.ensemble_save_dir:
            print(f"  ✓ Parameter merge successful")
        else:
            print(f"  ✗ Parameter merge failed")
            all_passed = False

    return all_passed


def test_check_existing_results():
    """Test check existing results"""
    print("\n" + "=" * 60)
    print("Testing check_existing_results")
    print("=" * 60)

    runner = EnsembleRunner(SCRIPTS_ROOT)

    for name in TEST_TOPK_NAMES[:2]:
        parsed = TSGymNameParser.parse(name)
        exists, result_dir = runner.check_existing_results(parsed.dataset, name)
        print(f"  Model: {name[:50]}...")
        print(f"    Exists: {exists}")
        print(f"    Result directory: {result_dir}")

    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# Starting Ensemble Module Tests")
    print("#" * 60)

    results = {}

    # Test 1: Name parsing
    results['name_parser'] = test_name_parser()

    # Test 2: Script retrieval
    results['find_script'] = test_find_script()

    # Test 3: Script parsing
    results['script_parser'] = test_script_parser()

    # Test 4: Check existing results
    results['check_existing'] = test_check_existing_results()

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ Passed" if passed else "✗ Failed"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "#" * 60)
    if all_passed:
        print("# All tests passed!")
    else:
        print("# Some tests failed!")
    print("#" * 60)

    return all_passed


if __name__ == '__main__':
    run_all_tests()