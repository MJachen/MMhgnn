param(
    [ValidateSet("0", "0.05", "0.1")]
    [string]$LambdaSeg = "0.05",
    [string]$Python = "python"
)

$configs = @{
    "0" = "configs/experiments/aux_wt_seg_lambda000.yaml"
    "0.05" = "configs/experiments/aux_wt_seg_lambda005.yaml"
    "0.1" = "configs/experiments/aux_wt_seg_lambda010.yaml"
}

& $Python train.py --config $configs[$LambdaSeg]
exit $LASTEXITCODE
