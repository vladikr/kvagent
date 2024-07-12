from langchain.chains.query_constructor.base import AttributeInfo

instTypes_document_content_description = "desciption of the virtual machine instance types"

instTypes_metadata_field_info = [
    AttributeInfo(
        name="cpu",
        description="Specifies the number of virtual CPUs the virtual machines requires",
        type="integer",
    ),
        AttributeInfo(
        name="dedicatedCPUPlacement",
        description="Indicates whether the virtual machines requires dedicated CPUs. dedicated CPUs also means excluse non-shared cpus. This is needed for high performance workloads",
        type="bool",
    ),
        AttributeInfo(
        name="isolateEmulatorThread",
        description="Indicates whether the virtual machines requires its emulator thread to be isolated on separate physcial CPU. This is needed for high performance workloads",
        type="bool",
    ),
        AttributeInfo(
        name="hugepages",
        description="Indicates whether the virtual machines requires hugepages. This is needed for higher performance",
        type="bool",
    ),
        AttributeInfo(
        name="memory",
        description="Specifies the amount of memory needed for the virtual machine to run. The value is in bytes.",
        type="integer",
    ),
        AttributeInfo(
        name="name",
        description="name of the virtual machine instance type",
        type="string",
    ),
        AttributeInfo(
        name="numa",
        description="Indicates whether the virtual machines guest NUMA topology should be mapped the hosts topology. This amy be needed for real time or latency sensitive workloads",
        type="bool",
    ),

]

