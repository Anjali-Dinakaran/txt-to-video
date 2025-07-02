def main_validation():
    """Main validation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate dataset")
    parser.add_argument("--data_dir", default="data/videos")
    parser.add_argument("--caption_file", default="data/captions.json")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess videos")
    parser.add_argument("--output_dir", default="data/processed_videos")
    
    args = parser.parse_args()
    
    # Validate dataset
    validator = DatasetValidator(args.data_dir, args.caption_file)
    is_valid = validator.validate()
    
    if args.stats:
        # Show statistics
        stats_generator = DatasetStatistics(args.data_dir, args.caption_file)
        stats_generator.print_stats()
    
    if args.preprocess and is_valid:
        # Preprocess videos
        preprocessor = VideoPreprocessor()
        preprocessor.preprocess_dataset(args.data_dir, args.output_dir, args.caption_file)
    
    return is_valid

if __name__ == "__main__":
    main_validation()