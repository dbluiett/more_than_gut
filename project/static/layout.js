
    var Component = require("./static/bower_components/pyxley/build/pyxley.js").FilterChart;
    var filter_style = "''";
var dynamic = true;
var charts = [{"type": "MetricsGraphics", "options": {"url": "/mghist/", "chart_id": "myhist", "params": {"right": 0, "target": "#myhist", "title": "Customer Persona Age", "buffer": 8, "small_width_threshold": 160, "top": 40, "bottom": 30, "height": 200, "width": 350, "chart_type": "histogram", "left": 0, "animate_on_load": "true", "small_height_threshold": 120, "init_params": {"persona": "John"}, "bins": 5, "description": "Histogram"}}}];
var filters = [{"type": "SelectButton", "options": {"default": "Jim", "items": ["Jim", "Joey", "Jack", "John"], "alias": "persona", "label": "Persona"}}];
    React.render(
        React.createElement(Component, {
        filter_style: filter_style, 
dynamic: dynamic, 
charts: charts, 
filters: filters}),
        document.getElementById("component_id")
    );
    