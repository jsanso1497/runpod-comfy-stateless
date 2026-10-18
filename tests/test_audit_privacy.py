import copy
import json
from workbench.export_privacy import public_verification


def test_nested_reference_report_cannot_bypass_metadata_opt_out():
    report={'outside_mask_changed_channels':0,'source_size':[200,300],
            'references':{'effective_prompt':'PRIVATE EDIT TEXT',
                          'references':[{'filename':'private_person.png','role':'my private role'}]},
            'mask':{'sam_prompt':'PRIVATE SAM TEXT','editable_pixels':42}}
    original=copy.deepcopy(report)
    clean=public_verification(report)
    assert clean=={'outside_mask_changed_channels':0,'source_size':[200,300],
                   'mask':{'editable_pixels':42}}
    assert report==original
    assert 'PRIVATE' not in json.dumps(clean) and 'private_person' not in json.dumps(clean)


def test_case_insensitive_nested_prompt_and_paths_are_removed():
    report={'audit':[{'Effective_Prompt':'hidden','PATH':'/private/person','count':3}],
            'value':True,'version':'fixed','source_url':'https://private.example'}
    clean=public_verification(report)
    assert clean=={'audit':[{'count':3}],'value':True,'version':'fixed'}
