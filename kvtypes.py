from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Dict
from typing_extensions import Literal
from typing_extensions import TypedDict

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

class Volume(BaseModel):
    name: str = Field(description="name of the volume, this name must match the corresponding disk name")
    #PersistentVolumeClaim: PersistentVolumeClaimVolumeSource
    containerDisk: ContainerDiskSource

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

class Metadata(BaseModel):
    name: str = Field(description="name of the virtual machine")
    labels: Dict[str, str]

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
