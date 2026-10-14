package responses

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolInvocationTextFilter_FullBlock(t *testing.T) {
	openTag := "<" + "tool_call>"
	closeTag := "</tool" + "_call>"
	rawJS := `text("hi");`
	block := openTag + rawJS + closeTag

	f := &toolInvocationTextFilter{}
	require.Empty(t, f.Filter(block))
	require.Empty(t, f.Flush())
}

func TestToolInvocationTextFilter_KeepsProseAroundBlock(t *testing.T) {
	openTag := "<" + "tool_call>"
	closeTag := "</tool" + "_call>"
	in := "before " + openTag + "payload" + closeTag + " after"

	f := &toolInvocationTextFilter{}
	out := f.Filter(in) + f.Flush()
	require.Equal(t, "before  after", out)
}

func TestToolInvocationTextFilter_SplitAcrossDeltas(t *testing.T) {
	parts := []string{"<", "tool_ca", "ll>const x=1;", "</tool_ca", "ll>"}

	f := &toolInvocationTextFilter{}
	var out strings.Builder
	for _, part := range parts {
		out.WriteString(f.Filter(part))
	}
	out.WriteString(f.Flush())
	require.Empty(t, out.String())
}

func TestToolInvocationTextFilter_FragmentAfterInterleavedToolCall(t *testing.T) {
	// Production shape: the rendering is split into two message items by a
	// tool-call delta; the tail starts mid-tag and must stay suppressed.
	openTag := "<" + "tool_call>"
	closeTag := "</tool" + "_call>"
	head := openTag + "const r=await tools.write_stdin({session_id:1,chars:\"\",yield"
	tail := "[parameter=yield_time_ms]330000[/parameter][/function" + "]" + closeTag

	f := &toolInvocationTextFilter{}
	require.Empty(t, f.Filter(head))
	require.Empty(t, f.Filter(tail))
	require.Empty(t, f.Flush())
}

func TestToolInvocationTextFilter_ParameterFragmentWithoutStart(t *testing.T) {
	// A tail fragment whose start tag lived in a previous message item. Once
	// the filter is inside the block it must swallow bare closing fragments.
	openTag := "<" + "tool_call>"
	tail := "[parameter=title]继续[/parameter][/function][/tool" + "_call]"

	f := &toolInvocationTextFilter{}
	require.Empty(t, f.Filter(openTag+"js"))
	require.Empty(t, f.Filter(tail))
}

func TestToolInvocationTextFilter_ProsePassthrough(t *testing.T) {
	f := &toolInvocationTextFilter{}
	require.Equal(t, "当前状态显示两个 pick 工作已部分完成。", f.Filter("当前状态显示两个 pick 工作已部分完成。"))
	require.Equal(t, "see tool docs", f.Filter("see tool docs"))
}

func TestToolInvocationTextFilter_DoesNotSwallowRestOnFalsePrefix(t *testing.T) {
	f := &toolInvocationTextFilter{}
	out := f.Filter("tool-call is not a marker")
	require.Equal(t, "tool-call is not a marker", out)
}

func TestStripToolInvocationText(t *testing.T) {
	openTag := "<" + "tool-use>"
	closeTag := "</parameter" + "></" + "function>"
	rawJS := `const r = await tools.exec_command({cmd:"ls"});`
	block := openTag + "\n" + rawJS + "\n" + closeTag

	require.Empty(t, StripToolInvocationText(block))
	require.Equal(t, "hello", StripToolInvocationText("hello"))
	require.Equal(t, "keep ", StripToolInvocationText("keep "+block))
}

func TestIsToolInvocationText(t *testing.T) {
	openTag := "<" + "tool-use>"
	require.True(t, IsToolInvocationText(openTag+"abc"))
	require.True(t, IsToolInvocationText("  "+openTag+"abc"))
	require.False(t, IsToolInvocationText("I will run the command."))
	require.False(t, IsToolInvocationText("see tool docs"))
}

func TestToolInvocationTextFilter_DropsOrphanTailAfterBlockClosed(t *testing.T) {
	// Production shape: the rendering head and tail land in separate message
	// items around a tool-call delta; a later tail can lose its head entirely
	// and must still be suppressed once a rendering has been seen.
	openTag := "<" + "tool_call>"
	closeTag := "</tool" + "_call>"
	head := openTag + `text(JSON.stringify(r));[/parameter]`
	tail := "[parameter=title]继续[/parameter][/function" + "]" + closeTag

	f := &toolInvocationTextFilter{}
	require.Empty(t, f.Filter(head)) // opens block, no close yet
	require.Empty(t, f.Filter(tail)) // closes block
	// Orphan tail whose opening tag lived in another message item.
	require.Empty(t, f.Filter("[parameter=yield_time_ms]330000[/parameter][/function" + "]" + closeTag))
	require.Empty(t, f.Filter("]>orphan[/function" + "]" + closeTag))
	require.Empty(t, f.Flush())
}

func TestToolInvocationTextFilter_KeepsProseAfterRendering(t *testing.T) {
	openTag := "<" + "tool_call>"
	closeTag := "</tool" + "_call>"
	f := &toolInvocationTextFilter{}
	require.Empty(t, f.Filter(openTag+"js"+closeTag))
	require.Equal(t, "随后继续分析。", f.Filter("随后继续分析。"))
}
