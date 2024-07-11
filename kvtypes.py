from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
from typing_extensions import Literal
from typing_extensions import TypedDict

class ResourceRequirements(BaseModel):
    class Config:
        extra = "forbid"

class StorageSpec(BaseModel):
    resources: ResourceRequirements
    class Config:
        extra = "forbid"

class DataVolumeMetadata(BaseModel):
    name: str = Field(description="name of the data volume, this name must match the corresponding volume name")
    labels: Optional[Dict[str, str]]

class DataVolumeSourceRef(BaseModel):
    kind: Literal["DataSource"]
    name: str = Field(description="name of the data volume source, this should be retrieved")
    namespace: Literal["openshift-virtualization-os-images"]

class DataVolumeSpec(BaseModel):
    sourceRef: DataVolumeSourceRef = Field(default=None, description="SourceRef is an indirect reference to the source of data for the requested DataVolume")
    storage: StorageSpec = Field(default=None, description="SourceRef is an indirect reference to the source of data for the requested DataVolume")

class DataVolumeTemplateSpec(BaseModel):
    spec: DataVolumeSpec
    metadata: DataVolumeMetadata

class BootSource(BaseModel):
  name: str = Field(
        description="Name of the data volume boot source.", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )

class RelatedBootSource(BaseModel):
    bootsources: List[BootSource] = Field(
        description="list of virtual machine boot sources",
        # Add a pydantic validation/restriction to be at most M editors
    )

class InstancetypeMatcher(BaseModel):
    name: str = Field(description="name of the virtual machine instance type")

class PreferenceMatcher(BaseModel):
    name: str = Field(description="name of the virtual machine preference")

class DiskTarget(BaseModel):
    class Config:
        extra = "forbid"

class Disk(BaseModel):
    name: str = Field(description="name of the disk, this name must match the corresponding volume name")
    disk: DiskTarget
    class Config:
        extra = "forbid"

class Devices(BaseModel):
    disks: List[Disk]

class DomainSpec(BaseModel):
    ## Devices allows adding disks, network interfaces, and others
    devices: Devices

class ContainerDiskSource(BaseModel):
    image: str = Field(description="image is the name of the image with the embedded disk")

class DataVolumeSource(BaseModel):
    name: str = Field(description="name of the corresponding dataVolume")

class Volume(BaseModel):
    name: str = Field(description="name of the volume, this name must match the corresponding disk name")
    #PersistentVolumeClaim: PersistentVolumeClaimVolumeSource
    #containerDisk: Optional[ContainerDiskSource]
    dataVolume: DataVolumeSource|None = None

class VirtualMachineInstanceSpec(BaseModel):
    domain: DomainSpec
    volumes: List[Volume]

class VirtualMachineInstanceTemplateSpec(BaseModel):
    spec: VirtualMachineInstanceSpec

class VirtualMachineSpec(BaseModel):
    running: bool = Field(False, cont=True)
    instancetype: InstancetypeMatcher
    preference: PreferenceMatcher
    template: VirtualMachineInstanceTemplateSpec
    dataVolumeTemplates: Optional[List[DataVolumeTemplateSpec]] = Field(default=None)#, exclude=True)

class Metadata(BaseModel):
    name: str = Field(description="name of the virtual machine")
    labels: Optional[Dict[str, str]]

class VirtualMachine(BaseModel):
    apiVersion: Literal['kubevirt.io/v1']
    kind: Literal["VirtualMachine"]
    metadata: Metadata|None = None
    spec: VirtualMachineSpec

class VmCreationState(TypedDict):
    definition: str
    virtualMachine: VirtualMachine
    complited: bool

class InstanceTypes(BaseModel):
  name: str = Field(
        description="Name of the instance type.", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )

class RelatedInstanceTypes(BaseModel):
    instanceTypes: List[InstanceTypes] = Field(
        description="list of virtual machine instance types",
        # Add a pydantic validation/restriction to be at most M editors
    )

class Preference(BaseModel):
  name: str = Field(
        description="Name of the preference.", pattern=r"^[a-zA-Z0-9_-]{1,64}$"
    )

class RelatedPreferences(BaseModel):
    preferences: List[Preference] = Field(
        description="list of virtual machine preferences",
        # Add a pydantic validation/restriction to be at most M editors
    )

class CallAgent(BaseModel):
    callAgent: str = Field(
        description="Select the next agent to run",
        # Add a pydantic validation/restriction to be at most M editors
    )

class RelatedVolumes(BaseModel):
    volumes: List[Volume] = Field(
        description="list of virtual machine volumes",
        # Add a pydantic validation/restriction to be at most M editors
    )
