<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1">
<meta name="generator" content="pdoc3 0.11.5">
<title>pytorch_symbolic.code_generator API documentation</title>
<meta name="description" content="">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/13.0.0/sanitize.min.css" integrity="sha512-y1dtMcuvtTMJc1yPgEqF0ZjQbhnc/bFhyvIyVNb9Zk5mIGtqVaAB1Ttl28su8AvFMOY0EwRbAe+HCLqj6W7/KA==" crossorigin>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/13.0.0/typography.min.css" integrity="sha512-Y1DYSb995BAfxobCkKepB1BqJJTPrOp3zPL74AWFugHHmmdcvO+C48WLrUOlhGMc0QG7AE3f7gmvvcrmX2fDoA==" crossorigin>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css" crossorigin>
<style>:root{--highlight-color:#fe9}.flex{display:flex !important}body{line-height:1.5em}#content{padding:20px}#sidebar{padding:1.5em;overflow:hidden}#sidebar > *:last-child{margin-bottom:2cm}.http-server-breadcrumbs{font-size:130%;margin:0 0 15px 0}#footer{font-size:.75em;padding:5px 30px;border-top:1px solid #ddd;text-align:right}#footer p{margin:0 0 0 1em;display:inline-block}#footer p:last-child{margin-right:30px}h1,h2,h3,h4,h5{font-weight:300}h1{font-size:2.5em;line-height:1.1em}h2{font-size:1.75em;margin:2em 0 .50em 0}h3{font-size:1.4em;margin:1.6em 0 .7em 0}h4{margin:0;font-size:105%}h1:target,h2:target,h3:target,h4:target,h5:target,h6:target{background:var(--highlight-color);padding:.2em 0}a{color:#058;text-decoration:none;transition:color .2s ease-in-out}a:visited{color:#503}a:hover{color:#b62}.title code{font-weight:bold}h2[id^="header-"]{margin-top:2em}.ident{color:#900;font-weight:bold}pre code{font-size:.8em;line-height:1.4em;padding:1em;display:block}code{background:#f3f3f3;font-family:"DejaVu Sans Mono",monospace;padding:1px 4px;overflow-wrap:break-word}h1 code{background:transparent}pre{border-top:1px solid #ccc;border-bottom:1px solid #ccc;margin:1em 0}#http-server-module-list{display:flex;flex-flow:column}#http-server-module-list div{display:flex}#http-server-module-list dt{min-width:10%}#http-server-module-list p{margin-top:0}.toc ul,#index{list-style-type:none;margin:0;padding:0}#index code{background:transparent}#index h3{border-bottom:1px solid #ddd}#index ul{padding:0}#index h4{margin-top:.6em;font-weight:bold}@media (min-width:200ex){#index .two-column{column-count:2}}@media (min-width:300ex){#index .two-column{column-count:3}}dl{margin-bottom:2em}dl dl:last-child{margin-bottom:4em}dd{margin:0 0 1em 3em}#header-classes + dl > dd{margin-bottom:3em}dd dd{margin-left:2em}dd p{margin:10px 0}.name{background:#eee;font-size:.85em;padding:5px 10px;display:inline-block;min-width:40%}.name:hover{background:#e0e0e0}dt:target .name{background:var(--highlight-color)}.name > span:first-child{white-space:nowrap}.name.class > span:nth-child(2){margin-left:.4em}.inherited{color:#999;border-left:5px solid #eee;padding-left:1em}.inheritance em{font-style:normal;font-weight:bold}.desc h2{font-weight:400;font-size:1.25em}.desc h3{font-size:1em}.desc dt code{background:inherit}.source > summary,.git-link-div{color:#666;text-align:right;font-weight:400;font-size:.8em;text-transform:uppercase}.source summary > *{white-space:nowrap;cursor:pointer}.git-link{color:inherit;margin-left:1em}.source pre{max-height:500px;overflow:auto;margin:0}.source pre code{font-size:12px;overflow:visible;min-width:max-content}.hlist{list-style:none}.hlist li{display:inline}.hlist li:after{content:',\2002'}.hlist li:last-child:after{content:none}.hlist .hlist{display:inline;padding-left:1em}img{max-width:100%}td{padding:0 .5em}.admonition{padding:.1em 1em;margin:1em 0}.admonition-title{font-weight:bold}.admonition.note,.admonition.info,.admonition.important{background:#aef}.admonition.todo,.admonition.versionadded,.admonition.tip,.admonition.hint{background:#dfd}.admonition.warning,.admonition.versionchanged,.admonition.deprecated{background:#fd4}.admonition.error,.admonition.danger,.admonition.caution{background:lightpink}</style>
<style media="screen and (min-width: 700px)">@media screen and (min-width:700px){#sidebar{width:30%;height:100vh;overflow:auto;position:sticky;top:0}#content{width:70%;max-width:100ch;padding:3em 4em;border-left:1px solid #ddd}pre code{font-size:1em}.name{font-size:1em}main{display:flex;flex-direction:row-reverse;justify-content:flex-end}.toc ul ul,#index ul ul{padding-left:1em}.toc > ul > li{margin-top:.5em}}</style>
<style media="print">@media print{#sidebar h1{page-break-before:always}.source{display:none}}@media print{*{background:transparent !important;color:#000 !important;box-shadow:none !important;text-shadow:none !important}a[href]:after{content:" (" attr(href) ")";font-size:90%}a[href][title]:after{content:none}abbr[title]:after{content:" (" attr(title) ")"}.ir a:after,a[href^="javascript:"]:after,a[href^="#"]:after{content:""}pre,blockquote{border:1px solid #999;page-break-inside:avoid}thead{display:table-header-group}tr,img{page-break-inside:avoid}img{max-width:100% !important}@page{margin:0.5cm}p,h2,h3{orphans:3;widows:3}h1,h2,h3,h4,h5,h6{page-break-after:avoid}}</style>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js" integrity="sha512-D9gUyxqja7hBtkWpPWGt9wfbfaMGVt9gnyCvYa+jojwwPHLCzUm5i8rpk7vD7wNee9bA35eYIjobYPaQuKS1MQ==" crossorigin></script>
<script>window.addEventListener('DOMContentLoaded', () => {
hljs.configure({languages: ['bash', 'css', 'diff', 'graphql', 'ini', 'javascript', 'json', 'plaintext', 'python', 'python-repl', 'rust', 'shell', 'sql', 'typescript', 'xml', 'yaml']});
hljs.highlightAll();
/* Collapse source docstrings */
setTimeout(() => {
[...document.querySelectorAll('.hljs.language-python > .hljs-string')]
.filter(el => el.innerHTML.length > 200 && ['"""', "'''"].includes(el.innerHTML.substring(0, 3)))
.forEach(el => {
let d = document.createElement('details');
d.classList.add('hljs-string');
d.innerHTML = '<summary>"""</summary>' + el.innerHTML.substring(3);
el.replaceWith(d);
});
}, 100);
})</script>
</head>
<body>
<main>
<article id="content">
<header>
<h1 class="title">Module <code>pytorch_symbolic.code_generator</code></h1>
</header>
<section id="section-intro">
</section>
<section>
</section>
<section>
</section>
<section>
<h2 class="section-title" id="header-functions">Functions</h2>
<dl>
<dt id="pytorch_symbolic.code_generator.generate_forward_with_loops"><code class="name flex">
<span>def <span class="ident">generate_forward_with_loops</span></span>(<span>inputs: List[SymbolicData] | Tuple[SymbolicData, ...],<br>outputs: List[SymbolicData] | Tuple[SymbolicData, ...],<br>execution_order: List[SymbolicData] | Tuple[SymbolicData, ...],<br>nodes_in_subgraph: Set[SymbolicData],<br>min_loop_length: int | float = inf) ‑> str</span>
</code></dt>
<dd>
<details class="source">
<summary>
<span>Expand source code</span>
</summary>
<pre><code class="python">def generate_forward_with_loops(
    inputs: List[SymbolicData] | Tuple[SymbolicData, ...],
    outputs: List[SymbolicData] | Tuple[SymbolicData, ...],
    execution_order: List[SymbolicData] | Tuple[SymbolicData, ...],
    nodes_in_subgraph: Set[SymbolicData],
    min_loop_length: int | float = float(&#34;inf&#34;),
) -&gt; str:
    &#34;&#34;&#34;Generate code for forward function of SymbolicModel.

    It assumes there is `self._execution_order_layers` available in the class.

    Parameters
    ----------
    inputs
        Inputs to the model
    outputs
        Outputs of the model
    execution_order
        Contains the exact order in which the nodes should be executed.
        If there are layers with multiple outputs, this will be a subset of `nodes_in_subgraph`.
        In such case, only one output of each layer needs to be in the execution_order.
    nodes_in_subgraph
        All nodes covered by the subgraph, including all nodes created by multiple-output layers.
    min_loop_length
        Minimal sequence length to replace sequential layers execution with a loop.

    Returns
    -------
    str
        Generated code.
    &#34;&#34;&#34;
    assert min_loop_length &gt;= 2, &#34;Loop length cannot be smaller than 2!&#34;

    str_length = len(str(max(len(inputs), len(outputs), len(execution_order))))
    node_to_name = {}
    for idx, node in enumerate(inputs):
        node_to_name[node] = f&#34;i{str(idx).zfill(str_length)}&#34;
    for idx, node in enumerate(execution_order):
        node_to_name[node] = f&#34;x{str(idx).zfill(str_length)}&#34;
    for idx, node in enumerate(nodes_in_subgraph.difference(execution_order)):
        node_to_name[node] = f&#34;y{str(idx).zfill(str_length)}&#34;
    for idx, node in enumerate(outputs):
        node_to_name[node] = f&#34;o{str(idx).zfill(str_length)}&#34;

    input_names = [node_to_name[node] for node in inputs]
    forward_definition = &#34;def forward(self,&#34; + &#34;, &#34;.join(input_names) + &#34;):&#34;
    code_lines = [forward_definition]

    TAB = &#34; &#34; * 4
    code_lines.append(TAB + &#34;l = self._execution_order_layers&#34;)

    nodes_looped_over = set()
    # All parents must be in the graph. Otherwise, forward is impossible.
    parents = {node: node.parents for node in execution_order}
    # We only count children in the graph. Thus the intersection.
    children = {node: list(nodes_in_subgraph.intersection(node.children)) for node in execution_order}

    siblings = {node: list(nodes_in_subgraph.intersection(node._layer_full_siblings)) for node in execution_order}

    for exec_id, node in enumerate(execution_order):
        if node in nodes_looped_over:
            continue

        input_names = [node_to_name[node] for node in node.parents]
        sequence = [node]
        last_node = node
        while (
            len(children[last_node]) == 1
            # stop iterating when need to unpack something!
            and len(last_node._layer_full_siblings) == 1
            and len(children[last_node][0]._layer_full_siblings) == 1  # needed, else tests fail
            # this should never be false, but just in case we make sure the child is next in execution order
            and children[last_node][0] is execution_order[exec_id + len(sequence)]
            and len(parents[last_node]) == 1
            and len(parents[children[last_node][0]]) == 1
        ):
            last_node = children[last_node][0]
            sequence.append(last_node)

        if len(sequence) &gt;= min_loop_length:
            output_name = node_to_name[sequence[-1]]
            code_lines.append(TAB + f&#34;{output_name} = {input_names[0]}&#34;)
            code_lines.append(TAB + f&#34;for layer in l[{exec_id}:{exec_id + len(sequence)}]:&#34;)
            code_lines.append(TAB + TAB + f&#34;{output_name} = layer({output_name})&#34;)
            nodes_looped_over.update(sequence)
        elif len(node._layer_full_siblings) &gt; 1:  # Must unpack all siblings, even if not all are used
            output_names = []
            for n in node._layer_full_siblings:
                if n in siblings[node]:
                    output_names.append(node_to_name[n])
                else:
                    output_names.append(&#34;_&#34;)  # If sibling not used, we don&#39;t save it as a variable

            assert len(input_names) == 1, &#34;Layer that has full siblings cannot have more than 1 input!&#34;
            code_line = TAB + &#34;, &#34;.join(output_names) + f&#34; = l[{exec_id}](&#34; + &#34;*&#34; + input_names[0] + &#34;)&#34;
            code_lines.append(code_line)
        else:
            code_line = TAB + node_to_name[node] + f&#34; = l[{exec_id}](&#34; + &#34;, &#34;.join(input_names) + &#34;)&#34;
            code_lines.append(code_line)

    code_lines.append(TAB + &#34;return &#34; + &#34;, &#34;.join(node_to_name[node] for node in outputs))
    generated_forward = &#34;\n&#34;.join(code_lines) + &#34;\n&#34;
    return generated_forward</code></pre>
</details>
<div class="desc"><p>Generate code for forward function of SymbolicModel.</p>
<p>It assumes there is <code>self._execution_order_layers</code> available in the class.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>inputs</code></strong></dt>
<dd>Inputs to the model</dd>
<dt><strong><code>outputs</code></strong></dt>
<dd>Outputs of the model</dd>
<dt><strong><code>execution_order</code></strong></dt>
<dd>Contains the exact order in which the nodes should be executed.
If there are layers with multiple outputs, this will be a subset of <code>nodes_in_subgraph</code>.
In such case, only one output of each layer needs to be in the execution_order.</dd>
<dt><strong><code>nodes_in_subgraph</code></strong></dt>
<dd>All nodes covered by the subgraph, including all nodes created by multiple-output layers.</dd>
<dt><strong><code>min_loop_length</code></strong></dt>
<dd>Minimal sequence length to replace sequential layers execution with a loop.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>str</code></dt>
<dd>Generated code.</dd>
</dl></div>
</dd>
</dl>
</section>
<section>
</section>
</article>
<nav id="sidebar">
<div class="toc">
<ul></ul>
</div>
<ul id="index">
<li><h3>Super-module</h3>
<ul>
<li><code><a title="pytorch_symbolic" href="index.html">pytorch_symbolic</a></code></li>
</ul>
</li>
<li><h3><a href="#header-functions">Functions</a></h3>
<ul class="">
<li><code><a title="pytorch_symbolic.code_generator.generate_forward_with_loops" href="#pytorch_symbolic.code_generator.generate_forward_with_loops">generate_forward_with_loops</a></code></li>
</ul>
</li>
</ul>
</nav>
</main>
<footer id="footer">
<p>Generated by <a href="https://pdoc3.github.io/pdoc" title="pdoc: Python API documentation generator"><cite>pdoc</cite> 0.11.5</a>.</p>
</footer>
</body>
</html>
