"""Exhaustive two-owner publication threshold boundary check through 400 frames."""
import json
import numpy as np
from baseline_bounded import O,reduce_surface_votes,CONFIRMED
from strict_votes import BASE
rows=[(a,b) for total in range(401) for a in range(total+1) for b in [total-a]]
keys=[];votes=[];expected=[]
for point,(a,b) in enumerate(rows):
 if a:keys.append(point*BASE+1);votes.append(a)
 if b:keys.append(point*BASE+2);votes.append(b)
 winner=1 if a>=b else 2
 expected.append(winner if max(a,b)>=2 and 100*max(a,b)>=67*(a+b) else -1)
value=reduce_surface_votes(np.asarray(keys,np.int64),np.asarray(votes,np.int32),len(rows),BASE,2,.67)
labels=np.where(value['state']==CONFIRMED,value['top1_instance_id'],-1)
np.testing.assert_array_equal(labels,np.asarray(expected))
report={'status':'PASS','count_pairs':len(rows),'effective_votes_range':[0,400],
 'independent_integer_rule':'winner_count >= 2 and 100 * winner_count >= 67 * total_effective_count',
 'unobserved_single_vote_ties_and_exact_67_percent_boundaries_checked':True}
(O/'publication_threshold_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
