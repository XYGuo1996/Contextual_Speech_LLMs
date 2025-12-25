import json
import os
import torch.utils.data as data_utils
from model.trainer import SpeechLLMLightning
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from model.dataset import MyCollator, InstructionalAudioDataset
import argparse
import distutils.util

def str2bool(v):
    return bool(distutils.util.strtobool(v))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=str, help='')
    parser.add_argument('--config', required=True, type=str, help='Config file path')
    parser.add_argument('--exp_dir', required=True, type=str, help='')
    parser.add_argument('--exp_name', required=True, type=str, help='Experiment name')
    parser.add_argument('--testsets', required=True, type=str, help='')
    parser.add_argument('--batch_size', required=True, type=int, help='')
    parser.add_argument('--decode_utt_by_utt', required=False, type=str2bool, default=False, help='Whether to decode utterance by utterance')
    parser.add_argument('--pre_utt_his_num', required=False, type=int, default=0, help='Number of previous utterances to use as history context')
 
    args, options = parser.parse_known_args()

    
    tests = args.testsets
    if "," in tests:
        tests = tests.split(",")
    else:
        tests = [tests]
    
    for testset in tests:
        basename = os.path.basename(testset)
        os.makedirs(os.path.join(args.exp_dir, args.exp_name, "infer"), exist_ok=True)
        
        with open(args.config) as f:
            model_config = json.load(f)
        model_config["exp_dir"] = args.exp_dir
        model_config["exp_name"] = args.exp_name
        model_config["test_basename"] = basename
        model_config["decode_utt_by_utt"] = args.decode_utt_by_utt
        model_config["pre_utt_his_num"] = args.pre_utt_his_num
        print(f"model_config['decode_utt_by_utt']: {model_config['decode_utt_by_utt']}")

        if args.decode_utt_by_utt:
            print(f"decode utt by utt")
            assert args.batch_size == 1, "When decode_utt_by_utt is True, batch_size must be 1."

        model = SpeechLLMLightning.load_from_checkpoint(
            args.ckpt, 
            model_config=model_config,
            num_validation_samples=0,
            strict=False
        )
        model.eval()

        my_collator = MyCollator(model.llm_tokenizer)

        adapter_name = model_config.get('adapter_name')

        test_dataset = InstructionalAudioDataset(
            json_file = testset,
            mode = 'test',
        )
        test_loader = data_utils.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=my_collator,
            num_workers=4
        )

        logger = CSVLogger(os.path.join(args.exp_dir, args.exp_name, "infer"), name=basename)
        trainer = Trainer(
            accelerator="auto",
            logger=logger,
            enable_checkpointing=False
        )

        test_results = trainer.test(model, dataloaders=test_loader)

        result_file = os.path.join(args.exp_dir, args.exp_name, "infer", f"{basename}_test_results.json")
        with open(result_file, "w") as f:
            json.dump(test_results, f, indent=4)

        print(f"please check: {result_file}")