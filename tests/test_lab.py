import json
import tempfile
import unittest
from pathlib import Path
import torch
from gmlp_lab.data import make_batch
from gmlp_lab.models import SpatialGatingUnit, GMLPBlock, GMLPClassifier, AttentionClassifier
from gmlp_lab.benchmark import fit_one


class SGUTests(unittest.TestCase):
    def test_shape(self):
        sg=SpatialGatingUnit(24,8)
        self.assertEqual(tuple(sg(torch.randn(3,8,24)).shape),(3,8,12))

    def test_short_sequence(self):
        sg=SpatialGatingUnit(16,8)
        self.assertEqual(tuple(sg(torch.randn(2,3,16)).shape),(2,3,8))

    def test_init(self):
        sg=SpatialGatingUnit(12,6)
        self.assertTrue(torch.allclose(sg.bias,torch.ones(6)))
        self.assertLess(float(sg.weight.detach().abs().max()),0.011)

    def test_gradients(self):
        sg=SpatialGatingUnit(12,6)
        z=torch.randn(2,6,12,requires_grad=True)
        sg(z).square().mean().backward()
        self.assertIsNotNone(sg.weight.grad)
        self.assertGreater(float(sg.weight.grad.abs().sum()),0.)

    def test_masked_zeros(self):
        sg=SpatialGatingUnit(12,6)
        with torch.no_grad():
            sg.weight.fill_(0.5)
        z=torch.randn(2,6,12)
        original=sg(z,torch.zeros(6,6))
        expected=z[:,:,:6]*sg.bias[None,:,None]
        self.assertTrue(torch.allclose(original,expected,atol=1e-6))

    def test_future_cannot_change_prefix(self):
        torch.manual_seed(42)
        block=GMLPBlock(8,16,9)
        with torch.no_grad():
            block.gate.weight.fill_(0.2)
        a=torch.randn(2,9,8)
        b=a.clone()
        b[:,5:]+=9
        causal=torch.tril(torch.ones(9,9))
        self.assertTrue(torch.allclose(block(a,causal)[:,:5], block(b,causal)[:,:5],atol=1e-6))

    def test_noncausal_future_changes_prefix(self):
        torch.manual_seed(42)
        block=GMLPBlock(8,16,9)
        with torch.no_grad():
            block.gate.weight.fill_(0.2)
        a=torch.randn(2,9,8)
        b=a.clone()
        b[:,5:]+=torch.randn_like(b[:,5:])*5
        self.assertGreater(float((block(a)[:,:5]-block(b)[:,:5]).detach().abs().max()),1e-4)

    def test_causal_gradient_no_future(self):
        torch.manual_seed(4)
        block=GMLPBlock(8,16,6)
        x=torch.randn(1,6,8,requires_grad=True)
        mask=torch.tril(torch.ones(6,6))
        block(x,mask)[:,2,:].sum().backward()
        self.assertLess(float(x.grad[:,3:].abs().max()),1e-7)

    def test_bad_masks(self):
        s=SpatialGatingUnit(12,6)
        with self.assertRaises(ValueError):
            s(torch.randn(2,6,12),torch.ones(5,5))

    def test_bad_dims(self):
        with self.assertRaises(ValueError):
            SpatialGatingUnit(11,6)
        with self.assertRaises(ValueError):
            GMLPBlock(8,15,6)

    def test_no_mixing_gradient(self):
        m=GMLPClassifier(disable_mixing=True)
        self.assertFalse(m.blocks[0].gate.weight.requires_grad)
        a=torch.randn(1,12,5)
        b=a.clone(); b[:,1:]+=30
        self.assertTrue(torch.allclose(m(a),m(b),atol=1e-6))

    def test_amlp_shape_and_mask(self):
        model=GMLPClassifier(tiny_attention_dim=16)
        x=torch.randn(2,12,5)
        self.assertEqual(tuple(model(x).shape),(2,2))
        causal=torch.tril(torch.ones(12,12))
        y=model.features(x,causal)
        edited=x.clone();edited[:,8:]=10
        y2=model.features(edited,causal)
        self.assertTrue(torch.allclose(y[:,:8],y2[:,:8],atol=1e-5))

    def test_shapes_models(self):
        x=torch.randn(4,12,5)
        self.assertEqual(tuple(GMLPClassifier()(x).shape),(4,2))
        self.assertEqual(tuple(AttentionClassifier()(x).shape),(4,2))


class DataTests(unittest.TestCase):
    def test_determinism(self):
        a=make_batch(6,12,'content',42)
        b=make_batch(6,12,'content',42)
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertTrue(torch.equal(a[1],b[1]))

    def test_content_labels(self):
        x,y=make_batch(60,12,'content',1)
        self.assertTrue(torch.equal(x[:,:,2].sum(1),torch.ones(60)))
        index=x[:,:,2].argmax(1)
        self.assertTrue(torch.equal(y,x[torch.arange(60),index,1].long()))

    def test_fixed_labels(self):
        x,y=make_batch(40,12,'fixed',2)
        self.assertTrue(torch.equal(y,x[:,3,1].long()))

    def test_holdout_changes(self):
        x,_=make_batch(60,12,'content',1)
        x2,_=make_batch(60,12,'content',2)
        self.assertFalse(torch.equal(x,x2))

    def test_marker_range(self):
        x,y=make_batch(100,12,'content',seed=33,marker_start=7,marker_end=12)
        positions=x[:,:,2].argmax(1)
        self.assertGreaterEqual(int(positions.min()),7)
        self.assertLess(int(positions.max()),12)

    def test_invalid_task(self):
        with self.assertRaises(ValueError):
            make_batch(5,12,'other',2)


class SmokeTests(unittest.TestCase):
    def test_one_step_training(self):
        m=GMLPClassifier(width=16,expansion=32,layers=1)
        optim=torch.optim.AdamW(m.parameters())
        x,y=make_batch(8,12,'fixed',1)
        loss=torch.nn.functional.cross_entropy(m(x),y)
        loss.backward(); optim.step()
        self.assertTrue(torch.isfinite(loss))

    def test_short_benchmark(self):
        result=fit_one('gmlp','fixed',seed=2,steps=2,batch_size=8,eval_size=16,width=16)
        self.assertGreaterEqual(result['accuracy'],0)
        self.assertLessEqual(result['accuracy'],1)

if __name__=='__main__':
    unittest.main()
