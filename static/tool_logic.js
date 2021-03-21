d3.json('/attrition_by_factor', function(data) {
    // Parse the json data to arrays:

    category = []
    count = []
    active = []
    turnover = []
    factors = []

    for (var i = 0; i < data.length; i++) {
        var cat = data[i]['Category']
        category.push(cat)
        var cnt = data[i]['Count']
        count.push(cnt)
        var act = data[i]['Active']
        active.push(act)
        var trn = data[i]['Terminated']
        turnover.push(trn)
        var fct = data[i]['factor']
        factors.push(fct)
    }

    // Populate the drop down menu
    d3.select("#selectFactor").selectAll()
        .data(factors)
        .enter()
        .append("option")
        .html(function(data) {
            return `<option>${data}</option>`
        })

    // Create event handler to trigger the table population function when selection is made
    input = d3.select("#selectFactor")
    input.on("change", function() {

        // reset form 
        d3.selectAll("tbody td").remove()
            // Filter data by user's selection and append it to table 
        var inputValue = d3.select("#selectFactor").node().value

        var i = factors.indexOf(inputValue)

        var table = d3.select("#table");
        var tbody = table.select("tbody");
        var trow;

        trow = tbody.append("tr");
        trow.append("td").text(factors[i]);
        trow.append("td").text(category[i][0]);
        trow.append("td").text(count[i][0]);
        trow.append("td").text(active[i][0])
        trow.append("td").text(turnover[i][0]);
        trow = tbody.append("tr")
        trow.append("td").text();
        trow.append("td").text(category[i][1]);
        trow.append("td").text(count[i][1]);
        trow.append("td").text(active[i][1]);
        trow.append("td").text(turnover[i][1]);
        trow = tbody.append("tr")
        trow.append("td").text();
        trow.append("td").text(category[i][2]);
        trow.append("td").text(count[i][2]);
        trow.append("td").text(active[i][2]);
        trow.append("td").text(turnover[i][2]);
        trow = tbody.append("tr")
        trow.append("td").text();
        trow.append("td").text(category[i][3]);
        trow.append("td").text(count[i][3]);
        trow.append("td").text(active[i][3]);
        trow.append("td").text(turnover[i][3]);
        trow = tbody.append("tr")
        trow.append("td").text();
        trow.append("td").text(category[i][4]);
        trow.append("td").text(count[i][4]);
        trow.append("td").text(active[i][4]);
        trow.append("td").text(turnover[i][4]);


        d3.json('/attrition_by_factor', function(trace_data) {
            d3.selectAll("#factorChart").html("")

            trace_data = [category[i], active[i], turnover[i]]

            var trace1 = {
                x: trace_data[0],
                y: trace_data[1],
                name: 'Retention',
                type: 'bar',
            };

            var trace2 = {
                x: trace_data[0],
                y: trace_data[2],
                name: 'Turnover',
                type: 'bar',
            }

            var chart_data = [trace1, trace2];
            var layout = {
                barmode: 'group',
                title: `${inputValue}`,
                yaxis: { title: "Percent" },
                legend: {
                    x: 0,
                    xanchor: 'left',
                    y: -.25
                },
                margin: {
                    t: 25,
                },
            };

            Plotly.newPlot("factorChart", chart_data, layout);
        })
    })

})

d3.json("/resources", function(table) {
    var table = document.querySelector("#tableArea > div.col-md-5.bls > table.dataframe > tbody")
    var row = table.rows

    for (var j = 0; j < row.length; j++) {

        row[j].deleteCell(0)
        row[j].deleteCell(0)
    }

    var headRow = document.querySelector("#tableArea > div.col-md-5.bls> table.dataframe > thead > tr")

    headRow.deleteCell(0)
    headRow.deleteCell(0)

})

d3.json('/turnover_cost', function(data) {
    var turnover = (Object.values(data[5]))
    var avg_cost = data[2].avg_cost
    var total_cost = data[3].total_cost
    var total_display_cost = total_cost.toLocaleString('en-US')
    var total_employees = data[0].total_employees

    d3.select("#currentRate").text(`Current Turnover Rate:  ${turnover}%`)
    d3.select("#currentCost").text(`Estimated Cost of Current Turnover:  $${total_display_cost}`)

    input = d3.select("#new_rate")
    input.on("change", runEstimate)


    function runEstimate() {
        var new_rate = (d3.select("#new_rate").node().value / 100)
        var new_cost = (avg_cost * (new_rate * total_employees))
        var difference = (parseInt(total_cost - new_cost))
        var post = difference.toLocaleString('en-US')
        d3.select("#cost_diff").text(`$${post}`)
        d3.select("#new_rate").node().value = "";
    }
})

d3.json('/employee_data', function(data) {
    input = d3.select("#emp_num")
    input.on("change", runPredict)

    function runPredict() {
        inputted_num = d3.select('#emp_num').node().value;
        const found = data.some(el => el.employee_number == inputted_num);
        if (!found) prediction = "Number not found";

        var prediction
        for (var i = 0; i < data.length; i++) {
            if (inputted_num == data[i].employee_number) {
                prediction = (data[i].loss_probability).toFixed(1);
            }
        }
        d3.select("#prediction").text(`${prediction}`);
        d3.select("#emp_num").node().value = "";

    }

})