.venv-exp/bin/python ./exp/mla_santapp.py \
    --prompt "Explain why the sky is blue." \
    --prompt-tokens 512 \
    --new-tokens 32 \
    --cluster-size 16 \
    --token-budget 128 \
    --recent-window 64
