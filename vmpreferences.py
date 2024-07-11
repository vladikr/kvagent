from langchain.chains.query_constructor.base import AttributeInfo
# Description of the loads prefs


prefs_document_content_description = "description of the Virtual Machine Preferences"

prefs_metadata_field_info = [
    AttributeInfo(
        name="clock.preferredTimer.hpet.present",
        description="Indicates whether HPET timer is required",
        type="bool",
    ),
        AttributeInfo(
        name="firmware.preferredUseSecureBoot",
        description="Indicated whether secure boot should be used",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="devices.preferredNetworkInterfaceMultiQueue",
        description="optionally enables the vhost multiqueue feature for virtio interfaces",
        type="bool",
    ),
        AttributeInfo(
        name="devices.preferredInputBus",
        description="optionally defines the preferred bus for Input devices",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="name",
        description="The name of the virtual machine preference",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="kind",
        description="The kind of the virtual machine preference",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="devices.preferredInterfaceModel",
        description="optionally defines the preferred model to be used by Interface devices",
        type="string or list[string]",
    ),
        AttributeInfo(
        name="preferredTerminationGracePeriodSeconds",
        description="Grace period observed after signalling a VirtualMachineInstance to stop after which the VirtualMachineInstance is force terminated",
        type="integer",
    ),
        AttributeInfo(
        name="devices.preferredDiskDedicatedIoThread",
        description="optionally enables dedicated IO threads for Disk devices",
        type="bool",
    ),
        AttributeInfo(
        name="requirements.cpu.guest",
        description="indicates the minimum number of CPUs required for this preference",
        type="integer",
    ),
        AttributeInfo(
        name="requirements.memory.guest",
        description="indicates the minimum memory size in bytes required for this preference",
        type="integer",
    ),
        AttributeInfo(
        name="firmware.preferredUseEfi",
        description="Indicated whether the EFI boot should be enabled",
        type="bool",
    ),]

