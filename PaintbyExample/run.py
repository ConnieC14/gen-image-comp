import os
import subprocess

if __name__ == "__main__":

    outdir = "results"
    config_path = "configs/v1.yaml"
    checkpoint_path = "checkpoints/model.ckpt"
    gt_path = "test_bench/GT_3500"
    mask_path = "test_bench/Mask_bbox_3500"
    ref_path = "test_bench/Ref_3500"
    seed = 0
    scale = 5

    results_dict = {}

    num_images = 2 # TODO change back to 10 - using 2 for testing small sample

    early_stop_counter = 0 # TODO delete - this is for testing on small sample
    
    for ref_file in os.listdir(ref_path):

        for i in range(1, num_images+1):
            image_name = str(i).zfill(12) + ".png"
            mask_image_path = os.path.join(mask_path, str(i).zfill(12) + "_mask.png")
            gt_image_path = os.path.join(gt_path, image_name)
            ref_image_name = ref_file #"chrome.png"  # TODO Example; adjust based on actual ref images if they vary
            ref_image_path = os.path.join(ref_path, ref_image_name)
            
            command = [
                "python", "scripts/inference.py",
                "--plms",
                "--outdir", outdir,
                "--config", config_path,
                "--ckpt", checkpoint_path,
                "--image_path", gt_image_path,
                "--mask_path", mask_image_path,
                "--reference_path", ref_image_path,
                "--seed", str(seed),
                "--scale", str(scale)
            ]

            print(f"Running command: {' '.join(command)}")
            
            # Execute the command
            subprocess.run(command, check=True, stderr=subprocess.STDOUT)
            
            result_image_name = f"{str(i).zfill(12)}_{ref_image_name}"  # This is an example, adjust based on actual naming
            
            # Update the results dictionary
            results_dict[result_image_name] = (mask_image_path, ref_image_path)

        early_stop_counter += 1

        if early_stop_counter >= 2:
            break

    # You can now use results_dict as needed
    print("Results:")
    print(results_dict)