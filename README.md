
### CLIP Score Evaluation

To run clip, first generate a id_list.npy file (replace X with the number of images generating):
    python create_id_list.py --count X

Then run the CLIP evaluation:
    python PaintbyExample/eval_tool/clip_score/region_clip_score.py --result_dir results