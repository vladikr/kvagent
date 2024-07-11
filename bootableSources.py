import os
import yaml
from kubernetes import client, config
from openshift.dynamic import DynamicClient
from langchain.chains.query_constructor.base import AttributeInfo
from openshift.helper.userpassauth import OCPLoginConfiguration


def get_client():
    server = "%s" % os.environ.get('K_SERVER')
    api_key = "%s" % os.environ.get('K_API_KEY')
    c = OCPLoginConfiguration()
    try:
        k8s_client = config.new_client_from_config()
    except:
        pass

    c.verify_ssl = False
    if server:
        c.host = server
    if api_key:
        c.api_key = {"authorization": "Bearer " + api_key}
 
    k8s_client = client.ApiClient(c)
    return DynamicClient(k8s_client)
    

def find_bootable_sources():
    client = get_client()
    datasources = client.resources.get(api_version='cdi.kubevirt.io/v1beta1', kind='DataSource')
    pvcs = client.resources.get(api_version='v1', kind='PersistentVolumeClaim')
    preferences = client.resources.get(api_version='instancetype.kubevirt.io/v1beta1', kind='VirtualMachinePreference')

    res = []

    for ds in datasources.get(namespace='openshift-virtualization-os-images').items:
        status = ds.get("status")
        if status:
            condtions = status.get('conditions')
            if condtions and len([x for x in condtions if x.get('type')=='Ready' and x.get("status")=='True']) > 0:
                pref = None
                name = ds['metadata'].get('name')

                dslabels = ds['metadata'].get('labels')
                if dslabels:
                    pref = ds['metadata']['labels'].get("instancetype.kubevirt.io/default-preference")
                if pref is None:
                    pvc = pvcs.get(namespace='openshift-virtualization-os-images', field_selector='metadata.name=%s' % name).items[0]
                    if pvc:
                        labels = pvc['metadata'].get('labels')
                        if labels:
                            pref = labels.get("instancetype.kubevirt.io/default-preference")
                preff = preferences.get(field_selector='metadata.name=%s' % pref).items[0]
                if preff:
                    dName = preff["metadata"]['annotations'].get("openshift.io/display-name")
                    if dName:
                        doc = dict()
                        doc['description'] = dName
                        doc['name'] = name
                        res.append(doc)
    return res

bootSrcs_document_content_description = "description or a full name of an operating system that can be used to boot a Virtual"

bootSrcs_metadata_field_info = [
        AttributeInfo(
        name="name",
        description="The name of the data source to use for booting a virtual machine",
        type="string",
    ),
]
