import os
import yaml
from kubernetes import client, config
from openshift.dynamic import DynamicClient

def get_client():
    server = os.environ.get('K_SERVER')
    api_key = os.environ.get('K_API_KEY')
    k8s_client = config.new_client_from_config()
    cc = k8s_client.configuration.get_default_copy()
    cc.verify_ssl = False
    if server:
        cc.host = server
    if api_key:
        cc.api_key = {"authorization": "Bearer " + api_key}
    k8s_client.configuration.set_default(cc)
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
            if condtions and len ([x for x in condtions if x.get('type')=='Ready' and x.get("status")=='True']) > 0:
                pref = "none"
                name = ds['metadata'].get('name')
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

