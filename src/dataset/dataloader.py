import os
import glob


VALID_FEATURES = {'prnu', 'illumination', 'frequency'}
VALID_COMBINATIONS = [
    frozenset({'prnu', 'frequency', 'illumination'}),
    frozenset({'prnu', 'frequency'}),
    frozenset({'prnu', 'illumination'}),
    frozenset({'frequency', 'illumination'}),
]

class DataLoader:
    def __init__(self, mask_folder, prnu_folder, illumination_folder, frequency_folder,
                 categories, templates,
                 features=('prnu', 'illumination', 'frequency')):
        """
        Args:
            mask_folder:          Folder name for masks.
            prnu_folder:          Folder name for PRNU feature images.
            illumination_folder:  Folder name for Illumination feature images.
            frequency_folder:     Folder name for Frequency feature images.
            categories:           List of category sub-folder names.
            templates:            List of template sub-folder names.
            features:             Iterable of feature names to use. Must be a supported
                                  combination of: 'prnu', 'illumination', 'frequency'.
                                  Supported combinations:
                                    - ('prnu', 'frequency', 'illumination')  [default]
                                    - ('prnu', 'frequency')
                                    - ('prnu', 'illumination')
                                    - ('frequency', 'illumination')
        """
        features = frozenset(f.lower() for f in features)

        if not features.issubset(VALID_FEATURES):
            invalid = features - VALID_FEATURES
            raise ValueError(f"Unknown feature(s): {invalid}. Valid options: {VALID_FEATURES}")

        if features not in VALID_COMBINATIONS:
            raise ValueError(
                f"Unsupported feature combination: {set(features)}.\n"
                f"Supported combinations: {[set(c) for c in VALID_COMBINATIONS]}"
            )

        self.features             = features
        self.TEMPLATES            = templates
        self.CATEGORIES           = categories
        self.MASK_FOLDER          = mask_folder
        self.PRNU_FOLDER          = prnu_folder
        self.FREQUENCY_FOLDER     = frequency_folder
        self.ILLUMINATION_FOLDER  = illumination_folder

        self._feature_folder = {
            'prnu':         self.PRNU_FOLDER,
            'illumination': self.ILLUMINATION_FOLDER,
            'frequency':    self.FREQUENCY_FOLDER,
        }

        print(f"[DataLoader] Active features: {sorted(self.features)}")

    def load_images(self, split: str, dataset_root: str):
        """
        Scans the dataset directory for a given split (Training or Validation) and returns a
        list of sample dicts:
            {
                'prnu':         path | None,
                'illumination': path | None,
                'frequency':    path | None,
                'mask':         path,
            }
        Only features listed in `self.features` are populated; others are None.
        Samples where any required feature file is missing are skipped with a warning.
        """
        samples = []
        missing_log = [] 
        split_root = os.path.join(dataset_root, split)

        for category in self.CATEGORIES:
            for template in self.TEMPLATES:
                mask_dir  = os.path.join(split_root, self.MASK_FOLDER,         category, template)

                # Build a dir path for each active feature
                feature_dirs = {
                    feat: os.path.join(split_root, self._feature_folder[feat], category, template)
                    for feat in self.features
                }

                # Use mask filenames as the reference
                if not os.path.isdir(mask_dir):
                    print(f'  WARNING: Directory not found — {mask_dir}')
                    continue

                for mask_path in sorted(glob.glob(os.path.join(mask_dir, '*'))):
                    if not os.path.isfile(mask_path):
                        continue

                    fname     = os.path.basename(mask_path)

                    # Build candidate paths only for active features
                    candidate_paths = {
                        feat: os.path.join(feat_dir, fname)
                        for feat, feat_dir in feature_dirs.items()
                    }
                    
                    missing = [p for p in candidate_paths.values() if not os.path.isfile(p)]
                    if missing:
                        missing_log.append((fname, missing))
                        continue

                    sample = {
                        'prnu':         candidate_paths.get('prnu',         None),
                        'illumination': candidate_paths.get('illumination', None),
                        'frequency':    candidate_paths.get('frequency',    None),
                        'mask':         mask_path,
                    }
                    samples.append(sample)

        if missing_log:
            print(f'[{split}] {len(missing_log)} samples skipped due to missing files:')
            for fname, missing in missing_log[:10]:  # show first 10
                print(f'  {fname} missing: {[os.path.basename(os.path.dirname(p)) for p in missing]}')
            if len(missing_log) > 10:
                print(f'  ... and {len(missing_log) - 10} more')

        print(f'[{split}] Total samples found: {len(samples)}')
        return samples
