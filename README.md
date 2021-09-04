# Face Anti-spoofing
> NTU DLCV(2020 Autumn) Final Project

## Intro.
Face anti-spoofing is the task of preventing false facial verification by using a phtot, video, mask or a different substitute for an authorized person's face. In this work, our solution is based on the method proposed by "*Single-Side Domain Generalization for Face Anti-Spoofing*" with some self-modifications added.

Report: [Report Link](https://github.com/vichsuWah/Face_AntiSpoofing/blob/main/Report.pdf)

## Usage
### To reproduce the challenges results
> python3 scripts/test_final.py --workspace \<testing folder path\> --outfile \<outfile folder+name\>
> 
> ex: python3 scripts/test_final.py --workspace data/oulu_npu_cropped/test --outfile ./oulu.csv
### To reproduce the bonus result
> python3 scripts/test_bonus.py --workspace \<testing folder path\> --outfile \<outfile folder+name\> 
> 
> ex: python scripts/test_bonus.py --workspace data/siw_test --outfile ./bonus.csv
---