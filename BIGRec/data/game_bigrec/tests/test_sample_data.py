import unittest
import os
import shutil
import json
import pandas as pd
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent dir to path to import sample_data if needed, 
# but we will likely run it via subprocess to test full CLI flow.
import subprocess

class TestSampleData(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'BIGRec/data/game_bigrec/tests/tmp_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.N = 100
        self.ItemNum = 50
        self.EmbDim = 16
        
        # Create Dummy train.json
        self.json_path = os.path.join(self.test_dir, 'train.json')
        self.data_json = []
        for i in range(self.N):
            self.data_json.append({
                "instruction": "test",
                "input": "test",
                "output": f"item_{i}",
                "meta": {"uid": i}
            })
        with open(self.json_path, 'w') as f:
            json.dump(self.data_json, f)
            
        # Create Dummy train_data.df
        self.df_path = os.path.join(self.test_dir, 'train_data.df')
        # Assign 'next' items. 
        # Make item 0 common, item 1 rare.
        targets = [0] * (self.N // 2) + [1] * (self.N // 4) + [i % self.ItemNum for i in range(self.N - 3 * (self.N // 4))]
        # Ensure targets match length N
        targets = targets[:self.N]
        while len(targets) < self.N:
            targets.append(0)
            
        df = pd.DataFrame({
            'uid': range(self.N),
            'next': targets,
            'seq': [[0,1]] * self.N,
            'len_seq': [2] * self.N
        })
        df.to_pickle(self.df_path)
        
        # Create Dummy train.pt (Logits)
        self.score_path = os.path.join(self.test_dir, 'train.pt')
        # Make UID 0 have correct prediction (low loss), UID 1 have wrong prediction (high loss)
        logits = torch.randn(self.N, self.ItemNum) * 0.1
        # UID 0: Target is 0. Make logit[0] high.
        logits[0, 0] = 10.0
        # UID 1: Target is 0. Make logit[0] low, sum/others high.
        logits[1, 0] = -10.0
        logits[1, 1] = 10.0 # Wrong prediction
        torch.save(logits, self.score_path)
        
        # Create Dummy train_uids.pt
        self.uid_path = os.path.join(self.test_dir, 'train_uids.pt')
        torch.save(torch.arange(self.N), self.uid_path)
        
        # Create Dummy embeddings.pt
        self.emb_path = os.path.join(self.test_dir, 'all_embeddings.pt')
        torch.save(torch.randn(self.ItemNum, self.EmbDim), self.emb_path)
        
        self.script_path = 'BIGRec/data/game_bigrec/sample_data.py'

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def run_script(self, method, sample_num, output_name, al_ratio=1.0):
        output_path = os.path.join(self.test_dir, output_name)
        cmd = [
            'python3', self.script_path,
            '--input_json', self.json_path,
            '--input_df', self.df_path,
            '--dros_score', self.score_path,
            '--dros_uid', self.uid_path,
            '--item_emb', self.emb_path,
            '--method', method,
            '--sample_num', str(sample_num),
            '--al_ratio', str(al_ratio),
            '--output_json', output_path,
            '--batch_size', '10'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        self.assertEqual(result.returncode, 0)
        return output_path

    def test_random(self):
        sample_num = 50
        output_path = self.run_script('random', sample_num, 'out_random.json')
        with open(output_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), sample_num)

    def test_loss(self):
        # We expect high loss items to be picked.
        # UID 1 has high loss (wrong prediction). UID 0 has low loss.
        # Sample 10 items. UID 1 should be there.
        sample_num = 10
        output_path = self.run_script('loss', sample_num, 'out_loss.json')
        with open(output_path, 'r') as f:
            data = json.load(f)
        uids = [d['meta']['uid'] for d in data]
        self.assertIn(1, uids)
        self.assertNotIn(0, uids)

    def test_clustering(self):
        sample_num = 50
        output_path = self.run_script('clustering', sample_num, 'out_cluster.json')
        with open(output_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), sample_num)

    def test_pop_inverse(self):
        sample_num = 20
        output_path = self.run_script('pop_inverse', sample_num, 'out_pop.json')
        with open(output_path, 'r') as f:
            data = json.load(f)
        # Should contain rare items. Item 1 is rare? No, item "remainder" are uniform rare.
        # Item 0 is popular. Should NOT be in top 20%.
        uids = [d['meta']['uid'] for d in data]
        # Most of picked uids should NOT have target 0
        df = pd.read_pickle(self.df_path)
        pick_targets = df[df['uid'].isin(uids)]['next'].tolist()
        # Item 0 count should be low/zero
        self.assertTrue(pick_targets.count(0) < len(pick_targets) * 0.5)

    def test_mixed(self):
        # UID 1 has high loss (wrong). UID 0 has low loss.
        # We want 50% AL, 50% Random.
        # Sample=10, AL=5. UID 1 must be in top 5.
        # Total 10 items.
        sample_num = 10
        output_path = self.run_script('loss', sample_num, 'out_mixed.json', al_ratio=0.5)
        with open(output_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 10)
        uids = [d['meta']['uid'] for d in data]
        self.assertIn(1, uids) # 1 is High Loss (Hard), should be picked by AL part.

if __name__ == '__main__':
    unittest.main()
