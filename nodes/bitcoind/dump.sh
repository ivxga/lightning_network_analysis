#!/bin/bash

export filename=latest.zip
export fileid=13yv41oNQKaYozKWMqB4dJYUysQW3BT10

### WGET ###
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename 'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
