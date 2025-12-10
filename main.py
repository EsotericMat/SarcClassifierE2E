import argparse
import logging
from sarcasm_classifier.components.preprocess import Preprocess
from sarcasm_classifier.components.train import Trainer

logger = logging.getLogger(__name__)

def run_preprocessing():
    """Run data preprocessing pipeline"""
    logger.info('[-] Step 1: Data Preprocessing')
    processor = Preprocess()
    processor.run()
    logger.info('[✓] Preprocessing completed')

def run_training(target='sarcasm'):
    """Run model training pipeline"""
    logger.info(f'[-] Step 2: Model Training (target: {target})')
    trainer = Trainer()
    trainer.run(target=target)
    logger.info('[✓] Training completed')

def run_full_pipeline(target='sarcasm'):
    """Run complete ML pipeline"""
    run_preprocessing()
    run_training(target)
    logger.info('[✓] Full pipeline completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sarcasm Classifier Pipeline')
    parser.add_argument('--step', choices=['preprocess', 'train', 'full'],
                       default='full', help='Pipeline step to run')
    parser.add_argument('--target', default='sarcasm',
                       help='Target for training (default: sarcasm)')

    args = parser.parse_args()

    if args.step == 'preprocess':
        run_preprocessing()
    elif args.step == 'train':
        run_training(args.target)
    else:
        run_full_pipeline(args.target)
