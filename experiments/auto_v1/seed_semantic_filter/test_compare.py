import copy
import unittest
from compare import compare


def fixtures():
    rows=[]
    for i in range(1,91):
        rows.append({'group_id':f'G{i:03d}','original_track_id':i,'mask_id':f'M{i:04d}','frame':i,
                     'mask_sha256':str(i),'RGB_sha256':str(i),'exclude':i in [1,16],
                     'decision':'suggest_exclude' if i in [1,16] else 'review',
                     'sensitivity_high_excluded_fraction':{'0.7':1 if i in [1,16] else 0}})
    result={'records':rows,'source_manifest_sha256':'frozen','human_selection_used':False,'GT_used':False,
            'config':{'sensitivity_pixel_thresholds':[0.7],'sensitivity_mask_thresholds':[0.9]},
            'stats':{},'model':{},'palette':[],'frames':[]}
    human={'status':'confirmed','qa_only':False,'manifest_sha256':'frozen','revision':67,
           'included':[r for r in rows if r['original_track_id']%2==0],
           'excluded':[r for r in rows if r['original_track_id']%2==1]}
    clip=copy.deepcopy(result)
    for r in clip['records']:
        r.update(exclude=r['original_track_id']==1,margin=0.04,top_term='wall')
    return result,human,clip


class CompareTests(unittest.TestCase):
    def test_expected_partition_and_new_exclusion(self):
        result,human,clip=fixtures();data=compare(result,human,clip)
        self.assertEqual(data['counts'],{'both_excluded':1,'auto_only_excluded':1,'human_only_excluded':44,'both_retained':44})
        self.assertEqual(data['versus_clip']['semantic_only_excluded'],['G016'])
        self.assertFalse(data['manual_reason_labels_available'])

    def test_different_original_masks_rejected(self):
        result,human,clip=fixtures();clip['records'][0]['mask_sha256']='other'
        with self.assertRaises(ValueError): compare(result,human,clip)

    def test_different_rgb_rejected(self):
        result,human,clip=fixtures();clip['records'][0]['RGB_sha256']='other'
        with self.assertRaises(ValueError): compare(result,human,clip)

    def test_unconfirmed_or_contaminated_reference_rejected(self):
        for option in ['status','qa','leak','duplicate']:
            result,human,clip=fixtures()
            if option=='status': human['status']='draft'
            if option=='qa': human['qa_only']=True
            if option=='leak': result['human_selection_used']=True
            if option=='duplicate': human['included'].append(human['included'][0])
            with self.assertRaises(ValueError): compare(result,human,clip)


if __name__=='__main__': unittest.main()
