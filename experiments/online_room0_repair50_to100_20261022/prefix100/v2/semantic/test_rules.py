import copy
import unittest
import numpy as np
from rules import resolve_labels, summarize_mask, filtered_manifest

LABELS = {'0':'cabinet','1':'rug','2':'ceiling','3':'floor','4':'wall','5':'painting'}
CFG = {'exclude_classes':['wall','floor','ceiling','rug'], 'pixel_confidence':0.7,
       'mask_fraction':0.9, 'sensitivity_pixel_thresholds':[0.6,0.7,0.8]}


class SemanticRulesTests(unittest.TestCase):
    def summarize(self, ids, confidence=0.99, mask=None):
        ids = np.array(ids, dtype=np.int32).reshape(1,-1)
        conf = np.broadcast_to(np.asarray(confidence,dtype=np.float32),ids.shape).copy()
        return summarize_mask(ids, conf, np.ones(ids.shape,bool) if mask is None else mask, LABELS, CFG)

    def test_label_ids_come_from_model(self):
        self.assertEqual(resolve_labels(LABELS,CFG)[1],[4,3,2,1])

    def test_missing_or_duplicate_labels_rejected(self):
        for bad in [{**LABELS,'1':'carpet'},{**LABELS,'0':'wall'}]:
            with self.assertRaises(ValueError): resolve_labels(bad,CFG)

    def test_all_authorized_categories(self):
        for label in [1,2,3,4]: self.assertTrue(self.summarize([label]*10)['exclude'])

    def test_nonexcluded_objects_protected(self):
        for label in [0,5]: self.assertEqual(self.summarize([label]*10)['decision'],'retain')

    def test_mask_denominator_not_bbox_or_frame(self):
        row=self.summarize([0,4,0,4,0],mask=np.array([[0,1,0,1,0]],bool))
        self.assertEqual(row['area'],2)
        self.assertEqual(row['high_excluded_fraction'],1)

    def test_90_percent_boundary_inclusive(self):
        self.assertTrue(self.summarize([4]*9+[0])['exclude'])
        self.assertFalse(self.summarize([4]*8+[0]*2)['exclude'])

    def test_mixed_mask_remains_review(self):
        self.assertEqual(self.summarize([4]*51+[0]*49)['decision'],'review')

    def test_low_confidence_background_remains_review(self):
        row=self.summarize([4]*10,0.699)
        self.assertEqual(row['decision'],'review')
        self.assertEqual(row['excluded_fraction'],1)
        self.assertEqual(row['high_excluded_fraction'],0)

    def test_confidence_boundary_and_sensitivity(self):
        row=self.summarize([4]*10,0.7)
        self.assertTrue(row['exclude'])
        self.assertEqual(row['sensitivity_high_excluded_fraction']['0.8'],0)

    def test_total_and_category_counts(self):
        row=self.summarize([4,4,0,1,5],[[.9,.5,.8,.95,.2]])
        self.assertEqual(sum(x['pixels'] for x in row['categories']),5)
        self.assertEqual(row['high_excluded_pixels']+row['high_retained_pixels']+row['low_confidence_pixels'],5)

    def test_invalid_input_rejected(self):
        for labels,conf,mask in [(np.array([[9]]),np.array([[.9]]),np.array([[True]])),
                                 (np.array([[4]]),np.array([[np.nan]]),np.array([[True]])),
                                 (np.array([[4]]),np.array([[.9]]),np.array([[False]])),
                                 (np.array([[4]]),np.array([[.9]]),np.array([[1]]))]:
            with self.assertRaises(ValueError): summarize_mask(labels,conf,mask,LABELS,CFG)

    def test_filtered_manifest_preserves_masks_sparse_ids_and_uncertainty(self):
        source={'groups':[{'id':f'G{i:03d}','primary':{'id':f'M{i:04d}','frame':i,'mask_sha256':str(i),'area':10,'mask':f'masks/{i}.png'}} for i in range(1,4)]}
        original=copy.deepcopy(source)
        records=[{'group_id':g['id'],'original_track_id':i,'mask_id':g['primary']['id'],
                  'frame':i,'mask_sha256':str(i),'area':10,'exclude':i==2,
                  'decision':'suggest_exclude' if i==2 else 'review'} for i,g in enumerate(source['groups'],1)]
        out=filtered_manifest(source,records,'sourcehash','resultsha',CFG)
        self.assertEqual([g['original_track_id'] for g in out['groups']],[1,3])
        self.assertEqual(out['groups'][1]['primary'],source['groups'][2]['primary'])
        self.assertEqual(source,original)
        self.assertEqual(out['uncertain_group_ids'],['G001','G003'])
        records[2]['mask_sha256']='changed'
        with self.assertRaises(ValueError): filtered_manifest(source,records,'s','r',CFG)


if __name__=='__main__': unittest.main()
