# Set encoding to support Python 2
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import unittest
from os import path
import numpy as np
import fasttext as ft

cbow_file = path.join(path.dirname(__file__), 'cbow_params_test.bin')
input_file = path.join(path.dirname(__file__), 'params_test.txt')
output = path.join(path.dirname(__file__), 'generated_cbow')

# Test to make sure that cbow interface run correctly
class TestCBOWModel(unittest.TestCase):
    def test_load_cbow_model(self):
        model = ft.load_model(cbow_file)

        # Make sure the model is returned correctly
        self.assertEqual(model.model_name, 'cbow')

        # Make sure all params loaded correctly
        # see Makefile on target test-cbow for the params
        self.assertEqual(model.dim, 50)
        self.assertEqual(model.ws, 5)
        self.assertEqual(model.epoch, 1)
        self.assertEqual(model.min_count, 1)
        self.assertEqual(model.neg, 5)
        self.assertEqual(model.loss_name, 'ns')
        self.assertEqual(model.bucket, 2000000)
        self.assertEqual(model.minn, 3)
        self.assertEqual(model.maxn, 6)
        self.assertEqual(model.lr_update_rate, 100)
        self.assertEqual(model.t, 1e-4)

        # Make sure the vector have the right dimension
        self.assertEqual(len(model['the']), model.dim)

        # Make sure we support unicode character
        unicode_str = 'Καλημέρα'
        self.assertTrue(unicode_str in model.words)
        self.assertEqual(len(model[unicode_str]), model.dim)

    def test_create_cbow_model(self):
        # set params
        lr=0.005
        dim=10
        ws=5
        epoch=5
        min_count=1
        neg=5
        word_ngrams=1
        loss='ns'
        bucket=2000000
        minn=3
        maxn=6
        thread=4
        lr_update_rate=10000
        t=1e-4
        silent=0

        # train cbow model
        model = ft.cbow(input_file, output, lr, dim, ws, epoch, min_count,
                neg, word_ngrams, loss, bucket, minn, maxn, thread, lr_update_rate,
                t, silent)

        # Make sure the model is generated correctly
        self.assertEqual(model.dim, dim)
        self.assertEqual(model.ws, ws)
        self.assertEqual(model.epoch, epoch)
        self.assertEqual(model.min_count, min_count)
        self.assertEqual(model.neg, neg)
        self.assertEqual(model.loss_name, loss)
        self.assertEqual(model.bucket, bucket)
        self.assertEqual(model.minn, minn)
        self.assertEqual(model.maxn, maxn)
        self.assertEqual(model.lr_update_rate, lr_update_rate)
        self.assertEqual(model.t, t)

        # Make sure .bin and .vec are generated
        self.assertTrue(path.isfile(output + '.bin'))
        self.assertTrue(path.isfile(output + '.vec'))

        # Make sure the model contains the word "the"
        self.assertTrue("the" in model)

        # Make sure the vector have the right dimension
        self.assertEqual(len(model['the']), dim)
        self.assertEqual(len(model.get_vector('the')), dim)

        # Make sure L2 normalization is working as expected
        self.assertGreater(abs(np.linalg.norm(model['the']) - 1), 1e-5)
        model.set_vec_norm(True)
        self.assertLess(abs(np.linalg.norm(model['the']) - 1), 1e-5)
        model.set_vec_norm(False)

        # Make sure we support unicode character
        unicode_str = 'Καλημέρα'
        self.assertTrue(unicode_str in model.words)
        self.assertTrue(unicode_str in model)
        self.assertEqual(len(model[unicode_str]), model.dim)
        self.assertEqual(len(model.get_vector(unicode_str)), model.dim)

if __name__ == '__main__':
    unittest.main()
