/**
 * Created by mayur on 02-04-2017.
 */

  $("#random").hide();
  $("#stratified").hide();
  $("#stratifiedImages").hide();
  $("#randomImages").hide();

function drawbar()
{
    var margin = {top: 20, right: 20, bottom: 30, left: 40},
        width = 960 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

// set the ranges
    var x = d3.scaleBand()
        .range([0, width])
        .padding(0.1);
    var y = d3.scaleLinear()
        .range([height, 0]);

// append the svg object to the body of the page
// append a 'group' element to 'svg'
// moves the 'group' element to the top left margin
    var svg = d3.select("#Bar").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)

    var g = svg.append("g")
        .attr("transform",
            "translate(" + 150 + "," + 10 + ")");

// get the data
    d3.csv("./static/CSV/PCA_square.csv", function (error, data) {
        if (error) throw error;

        // format the data
        data.forEach(function (d) {
            d.squareloadings = +d.squareloadings;
        });

        // Scale the range of the data in the domains
        x.domain(data.map(function (d) {
            return d.features;
        }));
        y.domain([0, d3.max(data, function (d) {
            return d.squareloadings;
        })]);

        // append the rectangles for the bar chart
        g.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", function (d) {
                return x(d.features);
            })
            .attr("width", x.bandwidth())
            .attr("y", function (d) {
                return y(d.squareloadings);
            })
            .attr("height", function (d) {
                return height - y(d.squareloadings);
            });

        // add the x Axis
        g.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
                .append("text")
            .text("Features")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(" + (width + margin.left) / 2 + ", " + 25 +  ")");

        // add the y Axis
        g.append("g")
            .call(d3.axisLeft(y))
        .append("text")
            .text("Square Loadings")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(-30, " + (height + margin.bottom) / 2 +  ") rotate(-90)");

    });
}
function drawRandomBar()
{
    var margin = {top: 20, right: 20, bottom: 30, left: 40},
        width = 960 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

// set the ranges
    var x = d3.scaleBand()
        .range([0, width])
        .padding(0.1);
    var y = d3.scaleLinear()
        .range([height, 0]);

// append the svg object to the body of the page
// append a 'group' element to 'svg'
// moves the 'group' element to the top left margin
    var svg = d3.select("#Barrandom").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)

    var g = svg.append("g")
        .attr("transform",
            "translate(" + 150 + "," + 10 + ")");

// get the data
    d3.csv("./static/CSV/PCArandom_square.csv", function (error, data) {
        if (error) throw error;

        // format the data
        data.forEach(function (d) {
            d.squareloadings = +d.squareloadings;
        });

        // Scale the range of the data in the domains
        x.domain(data.map(function (d) {
            return d.features;
        }));
        y.domain([0, d3.max(data, function (d) {
            return d.squareloadings;
        })]);

        // append the rectangles for the bar chart
        g.selectAll(".bar")
            .data(data)
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", function (d) {
                return x(d.features);
            })
            .attr("width", x.bandwidth())
            .attr("y", function (d) {
                return y(d.squareloadings);
            })
            .attr("height", function (d) {
                return height - y(d.squareloadings);
            });

        // add the x Axis
        g.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
                .append("text")
            .text("Features")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(" + (width + margin.left) / 2 + ", " + 25 +  ")");

        // add the y Axis
        g.append("g")
            .call(d3.axisLeft(y))
        .append("text")
            .text("Square Loadings")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(-30, " + (height + margin.bottom) / 2 +  ") rotate(-90)");

    });
}

function drawKMeansPlot()
{
    // set the dimensions and margins of the graph
var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;


// set the ranges
var x = d3.scaleLinear().range([0, width]);
var y = d3.scaleLinear().range([height, 0]);

// define the line
var valueline = d3.line()
    .x(function(d) { return x(d.clusters); })
    .y(function(d) { return y(d.meandist); });

// append the svg obgect to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("#K").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + 150 + "," + 10 + ")");

// Get the data
d3.csv("./static/CSV/K.csv", function(error, data) {
  if (error) throw error;

  // format the data
  data.forEach(function(d) {
      d.clusters = +d.clusters;
      d.meandist = +d.meandist;
  });

  // Scale the range of the data
  x.domain([d3.min(data, function(d) { return d.clusters; }), d3.max(data, function(d) { return d.clusters; })]);
  y.domain([d3.min(data, function(d) { return d.meandist; }), d3.max(data, function(d) { return d.meandist; })]);

  // Add the valueline path.
  svg.append("path")
      .data([data])
      .attr("class", "line")
      .attr("d", valueline);

  // Add the scatterplot
  svg.selectAll("dot")
      .data(data)
    .enter().append("circle")
      .attr("r", 5)
      .attr("cx", function(d) { return x(d.clusters); })
      .attr("cy", function(d) { return y(d.meandist); });

  // Add the X Axis
  svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x))
          .append("text")
            .text("Clusters")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(" + (width + margin.left) / 2 + ", " + 25 +  ")");

  // Add the Y Axis
  svg.append("g")
      .call(d3.axisLeft(y))
    .append("text")
            .text("Mean distances")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(-30, " + (height + margin.bottom) / 2 +  ") rotate(-90)");

});
}


function drawEigenPlot()
{
    // set the dimensions and margins of the graph
var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;


// set the ranges
var x = d3.scaleLinear().range([0, width]);
var y = d3.scaleLinear().range([height, 0]);

// define the line
var valueline = d3.line()
    .x(function(d) { return x(d.Components); })
    .y(function(d) { return y(d.Eigen); });

// append the svg obgect to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("#Eigen").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + 150 + "," + 10 + ")");

// Get the data
d3.csv("./static/CSV/PCA_Eigen.csv", function(error, data) {
  if (error) throw error;

  // format the data
  data.forEach(function(d) {
      d.Components = +d.Components;
      d.Eigen = +d.Eigen;
  });

  // Scale the range of the data
  x.domain([d3.min(data, function(d) { return d.Components; }), d3.max(data, function(d) { return d.Components; })]);
  y.domain([d3.min(data, function(d) { return d.Eigen; }), d3.max(data, function(d) { return d.Eigen; })]);

  // Add the valueline path.
  svg.append("path")
      .data([data])
      .attr("class", "line")
      .attr("d", valueline);

  // Add the scatterplot
  svg.selectAll("dot")
      .data(data)
    .enter().append("circle")
      .attr("r", 5)
      .attr("cx", function(d) { return x(d.Components); })
      .attr("cy", function(d) { return y(d.Eigen); });

  // Add the X Axis
  svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x))
          .append("text")
            .text("Components")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(" + (width + margin.left) / 2 + ", " + 25 +  ")");

  // Add the Y Axis
  svg.append("g")
      .call(d3.axisLeft(y))
    .append("text")
            .text("Eigen Values")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(-30, " + (height + margin.bottom) / 2 +  ") rotate(-90)");

});
}


function drawRandomEigenPlot()
{
    // set the dimensions and margins of the graph
var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;


// set the ranges
var x = d3.scaleLinear().range([0, width]);
var y = d3.scaleLinear().range([height, 0]);

// define the line
var valueline = d3.line()
    .x(function(d) { return x(d.Components); })
    .y(function(d) { return y(d.Eigen); });

// append the svg obgect to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("#Eigenrandom").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + 150 + "," + 10 + ")");

// Get the data
d3.csv("./static/CSV/PCARandom_Eigen.csv", function(error, data) {
  if (error) throw error;

  // format the data
  data.forEach(function(d) {
      d.Components = +d.Components;
      d.Eigen = +d.Eigen;
  });

  // Scale the range of the data
  x.domain([d3.min(data, function(d) { return d.Components; }), d3.max(data, function(d) { return d.Components; })]);
  y.domain([d3.min(data, function(d) { return d.Eigen; }), d3.max(data, function(d) { return d.Eigen; })]);

  // Add the valueline path.
  svg.append("path")
      .data([data])
      .attr("class", "line")
      .attr("d", valueline);

  // Add the scatterplot
  svg.selectAll("dot")
      .data(data)
    .enter().append("circle")
      .attr("r", 5)
      .attr("cx", function(d) { return x(d.Components); })
      .attr("cy", function(d) { return y(d.Eigen); });

  // Add the X Axis
  svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x))
          .append("text")
            .text("Components")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(" + (width + margin.left) / 2 + ", " + 25 +  ")");

  // Add the Y Axis
  svg.append("g")
      .call(d3.axisLeft(y))
    .append("text")
            .text("Eigen Values")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(-30, " + (height + margin.bottom) / 2 +  ") rotate(-90)");

});
}

drawbar();
drawRandomBar();
drawKMeansPlot();
drawEigenPlot();
drawRandomEigenPlot();

 $("#stratImg").click(function(e) {
  $("#random").hide();
 $("#stratified").hide();
 $("#randomImages").hide();
 $("#stratifiedImages").show();

});
$("#randImg").click(function(e) {

 $("#random").hide();
 $("#stratified").hide();
 $("#stratifiedImages").hide();
 $("#randomImages").show();
});


queue()
    .defer(d3.json, "/games/pca")
    .defer(d3.json, "/games/mdsCorrelation")
    .defer(d3.json, "/games/mdsEuclidean")
    .defer(d3.json, "/games/scatterplotMatrix")
    .defer(d3.json, "/games/pca/random")
    .defer(d3.json, "/games/mdsCorrelation/random")
    .defer(d3.json, "/games/mdsEuclidean/random")
    .defer(d3.json, "/games/scatterplotMatrix/random")
    .await(makePlots);

function makePlots(error, json_games, json_games_MDSC, json_games_MDSE, json_games_scatterplot_matrix, json_games_random, json_games_MDSC_random, json_games_MDSE_random, json_games_scatterplot_matrix_random) {
//function makePlots(error, json_games_random, json_games_MDSC_random, json_games_MDSE_random, json_games_scatterplot_matrix_random) {
 //function makePlots(error, json_games_random, json_games_scatterplot_matrix_random) {
    //function makePlots(error, json_games, json_games_scatterplot_matrix) {
    //function makePlots(error, json_games, json_games_random) {
    draw(json_games,json_games_random, "PCA");
    draw(json_games_MDSC,json_games_MDSC_random,  "MDSC");
    draw(json_games_MDSE,json_games_MDSE_random, "MDSE");
    drawScatterplot(json_games_scatterplot_matrix, "PCAS")

    //draw(json_games_random, "PCA");
    //draw(json_games_MDSE_random, "MDSER");
    //draw(json_games_MDSC_random, "MDSCR");
    drawScatterplot(json_games_scatterplot_matrix_random, "PCASR");

    $("#random").hide();
    $("#stratified").show();

}


$("#strat").click(function(e) {

 $("#random").hide();
 $("#stratifiedImages").hide();
 $("#randomImages").hide();
 $("#stratified").show();
});

$("#rand").click(function(e) {


 $("#stratifiedImages").hide();
 $("#randomImages").hide();
 $("#stratified").hide();
 $("#random").show();
});


function draw(json_games,json_games_random, id) {


    {
        var data = json_games;
        var color = d3.scaleOrdinal(d3.schemeCategory10);
        var margin = {top: 30, right: 30, bottom: 30, left: 250},
            width = 1240 - margin.left - margin.right,
            height = 560 - margin.top - margin.bottom;

        var svg = d3.select("#" + id).append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)

        g = svg.append("g")
            .attr("transform", "translate(" + margin.left + "," + 20 + ")")
            .attr("class", "PCA");

        data.forEach(function (d) {
            d.Component1 = +d.Component1;
            d.Component2 = +d.Component2;
        });


        var x = d3.scaleLinear().range([0, width]).domain([d3.min(data, function (d) {
            return d.Component1;
        }), d3.max(data, function (d) {
            return d.Component1;
        })]);
        var y = d3.scaleLinear().range([0, height]).domain([d3.min(data, function (d) {
            return d.Component2;
        }), d3.max(data, function (d) {
            return d.Component2;
        })]);


        g.selectAll("dot")
            .data(data)
            .enter().append("circle")
            .attr("r", 5)
            .attr("cx", function (d) {
                return x(d.Component1);
            })
            .attr("cy", function (d) {
                return y(d.Component2);
            })
            .style("fill", "black");


        // Add the X Axis
        g.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
            .append("text")
            .text("Principal Component 1")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(" + (width + margin.left) / 2 + ", " + 25 + ")");

        // Add the Y Axis
        g.append("g")
            .call(d3.axisLeft(y))
            .append("text")
            .text("Principal Component 2")
            .style("fill", "black")
            .style("stroke", "black")
            .attr("transform", "translate(-30, " + (height + margin.bottom) / 2 + ") rotate(-90)");

        var data = json_games_random;
        var color = d3.scaleOrdinal(d3.schemeCategory10);
        var margin = {top: 30, right: 30, bottom: 30, left: 250},
            width = 1240 - margin.left - margin.right,
            height = 560 - margin.top - margin.bottom;


        g = svg.append("g")
            .attr("transform", "translate(" + margin.left + "," + 20 + ")")
            .attr("class", "PCA");

        data.forEach(function (d) {
            d.Component1 = +d.Component1;
            d.Component2 = +d.Component2;
        });


        var x = d3.scaleLinear().range([0, width]).domain([d3.min(data, function (d) {
            return d.Component1;
        }), d3.max(data, function (d) {
            return d.Component1;
        })]);
        var y = d3.scaleLinear().range([0, height]).domain([d3.min(data, function (d) {
            return d.Component2;
        }), d3.max(data, function (d) {
            return d.Component2;
        })]);


        g.selectAll("dot")
            .data(data)
            .enter().append("circle")
            .attr("r", 5)
            .attr("cx", function (d) {
                return x(d.Component1);
            })
            .attr("cy", function (d) {
                return y(d.Component2);
            })
            .style("fill", "#1abc9c")
            .style("stroke", "black") ;

    }

}


function drawScatterplot(json_games_scatterplot_matrix, id) {


data = json_games_scatterplot_matrix;
    var width = 960,
    size = 230,
    padding = 20;

var x = d3.scaleLinear()
    .range([padding / 2, size - padding / 2]);

var y = d3.scaleLinear()
    .range([size - padding / 2, padding / 2]);

var xAxis = d3.axisBottom()
    .scale(x)
    .ticks(6);

var yAxis = d3.axisLeft()
    .scale(y)
    .ticks(6);

var color = d3.scaleOrdinal(d3.schemeCategory10);

  var Features = {},
      datapointss = d3.keys(data[0]).filter(function(d) { return d !== "species"; }),
      n = datapointss.length;

  datapointss.forEach(function(datapoints) {
    Features[datapoints] = d3.extent(data, function(d) { return d[datapoints]; });
  });

  xAxis.tickSize(size * n);
  yAxis.tickSize(-size * n);

  var brush = d3.brush()
      .on("start", brushstart)
      .on("brush", brushmove)
      .on("end", brushend)
      .extent([[0,0],[size,size]]);

  var svg = d3.select("#" + id).append("svg")
      .attr("width", 1240)
      .attr("height", size * n + padding)
      .append("g")
      .attr("transform", "translate(" + 250 + "," + padding / 2 + ")");

  svg.append("rect")
    .attr("width", "60%")
    .attr("height", "100%")
    .attr("fill", "#b0bec5");

  svg.selectAll(".x.axis")
      .data(datapointss)
    .enter().append("g")
      .attr("class", "x axis")
      .attr("transform", function(d, i) { return "translate(" + (n - i - 1) * size + ",0)"; })
      .each(function(d) { x.domain(Features[d]); d3.select(this).call(xAxis); });

  svg.selectAll(".y.axis")
      .data(datapointss)
    .enter().append("g")
      .attr("class", "y axis")
      .attr("transform", function(d, i) { return "translate(0," + i * size + ")"; })
      .each(function(d) { y.domain(Features[d]); d3.select(this).call(yAxis); });

  var cell = svg.selectAll(".cell")
      .data(cross(datapointss, datapointss))
    .enter().append("g")
      .attr("class", "cell")
      .attr("transform", function(d) { return "translate(" + (n - d.i - 1) * size + "," + d.j * size + ")"; })
      .each(plot);

  // Titles for the diagonal.
  cell.filter(function(d) { return d.i === d.j; }).append("text")
      .attr("x", padding)
      .attr("y", padding)
      .attr("dy", ".71em")
      .text(function(d) { return d.x; });

  cell.call(brush);

  function plot(p) {
    var cell = d3.select(this);

    x.domain(Features[p.x]);
    y.domain(Features[p.y]);

    cell.append("rect")
        .attr("class", "frame")
        .attr("x", padding / 2)
        .attr("y", padding / 2)
        .attr("width", size - padding)
        .attr("height", size - padding);

    cell.selectAll("circle")
        .data(data)
      .enter().append("circle")
        .attr("cx", function(d) { return x(d[p.x]); })
        .attr("cy", function(d) { return y(d[p.y]); })
        .attr("r", 4)
        .style("fill", function(d) { return color(d.Component1); })
      .style("stroke", "red") ;
  }

  var brushCell;

  // Clear the previously-active brush, if any.
  function brushstart(p) {
    if (brushCell !== this) {
      d3.select(brushCell).call(brush.move, null);
      brushCell = this;
    x.domain(Features[p.x]);
    y.domain(Features[p.y]);
    }
  }

  // Highlight the selected circles.
  function brushmove(p) {
    var e = d3.brushSelection(this);
    svg.selectAll("circle").classed("hidden", function(d) {
      return !e
        ? false
        : (
          e[0][0] > x(+d[p.x]) || x(+d[p.x]) > e[1][0]
          || e[0][1] > y(+d[p.y]) || y(+d[p.y]) > e[1][1]
        );
    });
  }

  // If the brush is empty, select all circles.
  function brushend() {
    var e = d3.brushSelection(this);
    if (e === null) svg.selectAll(".hidden").classed("hidden", false);
  }

function cross(a, b) {
  var c = [], n = a.length, m = b.length, i, j;
  for (i = -1; ++i < n;) for (j = -1; ++j < m;) c.push({x: a[i], i: i, y: b[j], j: j});
  return c;
}
}




// set the dimensions and margins of the graph


