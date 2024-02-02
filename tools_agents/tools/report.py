from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


# The actual function to be executed by the LLM
def write_report(filename, html):
    with open(filename, 'w') as f:
        f.write(html)


# Tell the LLM what arguments it needs to use the tool
class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


# We use a Structured Tool to pass multiple arguments, with the standard Tool you can only pass one arg. Due to
# legacy decisions
write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema
)

