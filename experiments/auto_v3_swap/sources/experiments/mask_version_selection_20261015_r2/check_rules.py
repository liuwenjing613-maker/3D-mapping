from selection_rules import choose, noise_reason

def n(k, eligible=True, crop=False, noise=None):
    m, lid=map(int,k.split(':'))
    return dict(key=k,model=m,label=lid,eligible=eligible,touches_crop=crop,touches_frame=False,noise_reason=noise)
def e(a,b,near=False,contain=False):
    return dict(a=a,b=b,group_link=True,near=near,containment=contain)
count=0
def run(nodes,edges,expected):
    global count
    actual=[(g['selected'],g['reason']) for g in choose(nodes,edges)]
    assert actual==expected,(actual,expected)
    count+=1

run([n('0:1')],[],[('0:1','SINGLE_REMAINING')])
run([n('1:1',crop=True)],[],[('1:1','SINGLE_REMAINING')])
run([n('0:1'),n('1:1',crop=True)],[e('0:1','1:1',True)],[('0:1','CROPFORMERS_AGREE')])
triple=[e('0:1','1:1',True),e('0:1','2:1',True),e('1:1','2:1',True)]
run([n('0:1'),n('1:1'),n('2:1')],triple,[('0:1','THREE_AGREE')])
run([n('0:1'),n('1:1',crop=True),n('2:1')],[e('0:1','1:1'),e('1:1','2:1',True)],[('1:1','ONE_CROPFORMER_WITH_SAM')])
run([n('0:1'),n('1:1',crop=True)],[e('0:1','1:1',contain=True)],[(None,'WHOLE_PART_CONFLICT')])
run([n('0:1'),n('1:1',crop=True)],[e('0:1','1:1')],[(None,'UNRESOLVED_DISAGREEMENT')])
run([n('0:1'),n('1:1'),n('2:1',False)],[e('0:1','1:1',True),e('0:1','2:1',contain=True)],[(None,'WHOLE_PART_CONFLICT')])
run([n('0:1'),n('0:2'),n('1:1',False)],[e('0:1','1:1'),e('0:2','1:1')],[('0:1','SINGLE_REMAINING'),('0:2','SINGLE_REMAINING')])
run([n('0:1'),n('1:1'),n('2:1'),n('2:2')],triple+[e('0:1','2:2')],[('0:1','AGREED_MAIN_KEEP'),(None,'EXTRA_FRAGMENT_REVIEW')])
run([n('0:1'),n('1:1'),n('2:1'),n('2:2',noise='TINY_AREA_NOISE')],triple+[e('0:1','2:2',contain=True)],[('0:1','THREE_AGREE')])
run([n('0:1'),n('1:1'),n('2:1'),n('2:2')],triple+[e('0:1','2:2',contain=True)],[(None,'WHOLE_PART_CONFLICT')])
run([n('0:1'),n('1:1'),n('2:1')],[e('0:1','1:1'),e('0:1','2:1',True),e('1:1','2:1',True)],[(None,'MULTIPLE_SUPPORTED_VERSIONS')])
run([n('0:1',noise='TINY_AREA_NOISE')],[],[])
assert noise_reason(25,5,5)=='TINY_AREA_NOISE'
assert noise_reason(26,5,6) is None
assert noise_reason(100,2,50)=='THIN_LINE_NOISE'
assert noise_reason(101,2,51) is None
assert noise_reason(80,2.1,40) is None
assert noise_reason(30,5,6) is None
assert noise_reason(1100,25,44) is None
print(f'PASS: {count} decision cases and 7 noise threshold/compact small-object cases')
