import os
import glob

class DataLoader:
    def __init__(self, mask_folder, prnu_folder, illumination_folder, frequency_folder, categories, templates):
        self.TEMPLATES = templates
        self.CATEGORIES = categories
        self.MASK_FOLDER = mask_folder
        self.PRNU_FOLDER = prnu_folder
        self.FREQUENCY_FOLDER = frequency_folder
        self.ILLUMINATION_FOLDER = illumination_folder

    def load_images(self, split: str, dataset_root: str):
        """
        Scans the dataset directory for a given split (Train or Validation) and returns a list of
        matched file path tuples: (prnu_path, illu_path, freq_path, mask_path).

        Only includes samples where ALL four files exist and are valid files.
        Non-existent directories are skipped with a warning.
        """
        samples = []
        missing_log = [] 
        split_root = os.path.join(dataset_root, split)

        for category in self.CATEGORIES:
            for template in self.TEMPLATES:
                mask_dir  = os.path.join(split_root, self.MASK_FOLDER,         category, template)
                prnu_dir  = os.path.join(split_root, self.PRNU_FOLDER,         category, template)
                illu_dir  = os.path.join(split_root, self.ILLUMINATION_FOLDER, category, template)
                freq_dir  = os.path.join(split_root, self.FREQUENCY_FOLDER,    category, template)

                # Use mask filenames as the reference
                if not os.path.isdir(mask_dir):
                    print(f'  WARNING: Directory not found — {mask_dir}')
                    continue

                for mask_path in sorted(glob.glob(os.path.join(mask_dir, '*'))):
                    if not os.path.isfile(mask_path):
                        continue

                    fname     = os.path.basename(mask_path)
                    prnu_path = os.path.join(prnu_dir,  fname)
                    illu_path = os.path.join(illu_dir,  fname)
                    freq_path = os.path.join(freq_dir,  fname)

                    # if all(os.path.isfile(p) for p in [prnu_path, illu_path, freq_path]):
                    #     samples.append((prnu_path, illu_path, freq_path, mask_path))

                    missing = [p for p in [prnu_path, illu_path, freq_path] if not os.path.isfile(p)]
                    if missing:
                        missing_log.append((fname, missing))  # ← log it
                    else:
                        samples.append((prnu_path, illu_path, freq_path, mask_path))

        if missing_log:
            print(f'[{split}] {len(missing_log)} samples skipped due to missing files:')
            for fname, missing in missing_log[:10]:  # show first 10
                print(f'  {fname} missing: {[os.path.basename(os.path.dirname(p)) for p in missing]}')
            if len(missing_log) > 10:
                print(f'  ... and {len(missing_log) - 10} more')

        print(f'[{split}] Total samples found: {len(samples)}')
        return samples
