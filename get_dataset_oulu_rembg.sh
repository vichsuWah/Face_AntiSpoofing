
# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1N4e_iY4YaGYcSZIhT0D2qwk98wZzn43O' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N4e_iY4YaGYcSZIhT0D2qwk98wZzn43O" -O train_x_rembg.npy && rm -rf /tmp/cookies.txt
