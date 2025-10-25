// Neural Network Hardware Implementation
// Tiny Tapeout Compatible - Packed Arrays Only
`timescale 1ns / 1ps

module tt_um_mark28277 (
    input wire [7:0] ui_in, //(dedicated inputs - connected to the input switches)
    output wire [7:0] uo_out, //(dedicated outputs - connected to the 7 segment display)
    input wire [7:0] uio_in, //(IOs: Bidirectional input path)
    output wire [7:0] uio_out, //(IOs: Bidirectional output path)
    output wire [7:0] uio_oe, //(IOs: Bidirectional enable path (active high: 0=input, 1=output))
    input wire           ena, //(will go high when the design is enabled)
    input wire           clk, //(clock)
    input wire           rst_n //(reset_n - low to reset)
);

    // Input interface for Tiny Tapeout limited I/O
    wire reset;
    assign reset = ~rst_n; // Convert active-low to active-high reset

    // Neural network input (8-bit for Tiny Tapeout)
    wire [7:0] input_data;
    assign input_data = ui_in; // Use dedicated input directly

    // Conv2d Layer 0
    wire [7:0] conv_0_out;
    conv2d_layer conv_inst_0 (
        .clk(clk),
        .reset(reset),
        .input_data(input_data),
        .output_data(conv_0_out)
    );

    // ReLU Layer 1
    wire [7:0] relu_1_out;
    relu_layer relu_inst_1 (
        .clk(clk),
        .reset(reset),
        .input_data(conv_0_out),
        .output_data(relu_1_out)
    );

    // MaxPool2d Layer 2
    wire [7:0] maxpool_2_out;
    maxpool_layer maxpool_inst_2 (
        .clk(clk),
        .reset(reset),
        .input_data(relu_1_out),
        .output_data(maxpool_2_out)
    );

    // Linear Layer 3
    wire [7:0] linear_3_out;
    linear_layer linear_inst_3 (
        .clk(clk),
        .reset(reset),
        .input_data(maxpool_2_out),
        .output_data(linear_3_out)
    );

    // Final output signal
    wire [7:0] final_output;
    assign final_output = linear_3_out;

    // Output interface for Tiny Tapeout limited I/O
    reg [7:0] uo_out_reg;
    reg [7:0] uio_out_reg;
    reg [7:0] uio_oe_reg;

    always @(posedge clk) begin
        if (reset) begin
            uo_out_reg <= 8'b0;
            uio_out_reg <= 8'b0;
            uio_oe_reg <= 8'b0;
        end else if (ena) begin
            // Output final result to dedicated output
            uo_out_reg <= final_output;
            // Output inverted result to bidirectional output
            uio_out_reg <= ~final_output;
            // Set all IOs as outputs
            uio_oe_reg <= 8'hFF;
        end
    end

    assign uo_out = uo_out_reg;
    assign uio_out = uio_out_reg;
    assign uio_oe = uio_oe_reg;

endmodule

// Conv2d Layer with Actual Weights for Tiny Tapeout
module conv2d_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data,
    output wire [7:0] output_data
);

    // Weight and bias storage
    reg [31:0] weights [0:1][0:2][0:2][0:2]; // 2x3x3x3 weights
    reg [31:0] biases [0:1]; // 2 biases

    // Weight initialization from trained model
    initial begin
        weights[0][0][0][0] = 32'h0000ff77;
        weights[0][0][0][1] = 32'h0000001e;
        weights[0][0][0][2] = 32'h00000131;
        weights[0][0][1][0] = 32'h0000001a;
        weights[0][0][1][1] = 32'h0000ffa5;
        weights[0][0][1][2] = 32'h00000005;
        weights[0][0][2][0] = 32'h0000009b;
        weights[0][0][2][1] = 32'h0000ffd6;
        weights[0][0][2][2] = 32'h0000ff69;
        weights[0][1][0][0] = 32'h0000fef6;
        weights[0][1][0][1] = 32'h0000ff9a;
        weights[0][1][0][2] = 32'h0000001f;
        weights[0][1][1][0] = 32'h00000055;
        weights[0][1][1][1] = 32'h0000ff5b;
        weights[0][1][1][2] = 32'h0000fff3;
        weights[0][1][2][0] = 32'h000000b8;
        weights[0][1][2][1] = 32'h0000ffd6;
        weights[0][1][2][2] = 32'h0000ff6b;
        weights[0][2][0][0] = 32'h0000ff5e;
        weights[0][2][0][1] = 32'h0000fff8;
        weights[0][2][0][2] = 32'h0000006a;
        weights[0][2][1][0] = 32'h0000001c;
        weights[0][2][1][1] = 32'h0000ff5d;
        weights[0][2][1][2] = 32'h0000001a;
        weights[0][2][2][0] = 32'h0000006b;
        weights[0][2][2][1] = 32'h0000fffd;
        weights[0][2][2][2] = 32'h0000ff4c;
        weights[1][0][0][0] = 32'h0000ff65;
        weights[1][0][0][1] = 32'h0000ffd0;
        weights[1][0][0][2] = 32'h0000ff89;
        weights[1][0][1][0] = 32'h00000008;
        weights[1][0][1][1] = 32'h00000009;
        weights[1][0][1][2] = 32'h0000001c;
        weights[1][0][2][0] = 32'h0000006e;
        weights[1][0][2][1] = 32'h00000057;
        weights[1][0][2][2] = 32'h00000032;
        weights[1][1][0][0] = 32'h0000ff59;
        weights[1][1][0][1] = 32'h0000ffb5;
        weights[1][1][0][2] = 32'h0000ff67;
        weights[1][1][1][0] = 32'h0000ffd0;
        weights[1][1][1][1] = 32'h00000018;
        weights[1][1][1][2] = 32'h0000ffcb;
        weights[1][1][2][0] = 32'h0000ffb4;
        weights[1][1][2][1] = 32'h0000ff72;
        weights[1][1][2][2] = 32'h0000ffc6;
        weights[1][2][0][0] = 32'h0000fff1;
        weights[1][2][0][1] = 32'h0000ffde;
        weights[1][2][0][2] = 32'h0000ffb8;
        weights[1][2][1][0] = 32'h00000093;
        weights[1][2][1][1] = 32'h00000082;
        weights[1][2][1][2] = 32'h000000b2;
        weights[1][2][2][0] = 32'h0000005b;
        weights[1][2][2][1] = 32'h00000022;
        weights[1][2][2][2] = 32'h00000063;
        biases[0] = 32'h00000029;
        biases[1] = 32'h0000ffdd;
    end

    // Convolution computation
    reg [31:0] conv_accumulator;
    reg [7:0] output_reg;
    reg [2:0] kernel_counter;
    reg computation_done;

    always @(posedge clk) begin
        if (reset) begin
            conv_accumulator <= 32'b0;
            output_reg <= 8'd42; // Start with non-zero value
            kernel_counter <= 3'b0;
            computation_done <= 1'b0;
        end else begin
            if (!computation_done) begin
                // Real convolution using actual weights
                // For Tiny Tapeout, we'll compute a simplified 3x3 convolution
                case (kernel_counter)
                    3'b000: conv_accumulator <= ({24'b0, input_data} * weights[0][0][0][0][15:0]);
                    3'b001: conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][0][1][15:0]);
                    3'b010: conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][0][2][15:0]);
                    3'b011: conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][1][0][15:0]);
                    3'b100: conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][1][1][15:0]);
                    3'b101: conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][1][2][15:0]);
                    3'b110: conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][2][0][15:0]);
                    3'b111: begin
                        conv_accumulator <= conv_accumulator + ({24'b0, input_data} * weights[0][0][2][1][15:0]);
                        // Final computation with bias and saturation
                        if ((conv_accumulator >> 8) + biases[0][15:0] > 255) begin
                            output_reg <= 8'd255;
                        end else if ((conv_accumulator >> 8) + biases[0][15:0] < 0) begin
                            output_reg <= 8'd0;
                        end else begin
                            output_reg <= (conv_accumulator >> 8) + biases[0][15:0];
                        end
                        computation_done <= 1'b1;
                    end
                endcase
                kernel_counter <= kernel_counter + 1;
            end
        end
    end

    assign output_data = output_reg;

endmodule

// Linear Layer with Actual Weights for Tiny Tapeout
module linear_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data,
    output wire [7:0] output_data
);

    // Weight and bias storage
    reg [31:0] weights [0:9][0:31]; // 10x32 weights
    reg [31:0] biases [0:9]; // 10 biases

    // Weight initialization from trained model
    initial begin
        weights[0][0] = 32'h0000ffea;
        weights[0][1] = 32'h0000ffbd;
        weights[0][2] = 32'h0000ff8c;
        weights[0][3] = 32'h0000ff8d;
        weights[0][4] = 32'h0000000a;
        weights[0][5] = 32'h0000ffd8;
        weights[0][6] = 32'h0000ffaa;
        weights[0][7] = 32'h0000ffc4;
        weights[0][8] = 32'h0000ffe6;
        weights[0][9] = 32'h0000001b;
        weights[0][10] = 32'h00000019;
        weights[0][11] = 32'h0000ffb4;
        weights[0][12] = 32'h0000ffd9;
        weights[0][13] = 32'h0000ffc8;
        weights[0][14] = 32'h00000004;
        weights[0][15] = 32'h00000010;
        weights[0][16] = 32'h00000016;
        weights[0][17] = 32'h0000ffe3;
        weights[0][18] = 32'h0000ffde;
        weights[0][19] = 32'h00000031;
        weights[0][20] = 32'h00000024;
        weights[0][21] = 32'h00000039;
        weights[0][22] = 32'h0000004b;
        weights[0][23] = 32'h00000083;
        weights[0][24] = 32'h00000039;
        weights[0][25] = 32'h00000037;
        weights[0][26] = 32'h0000003f;
        weights[0][27] = 32'h00000056;
        weights[0][28] = 32'h0000000f;
        weights[0][29] = 32'h0000fff8;
        weights[0][30] = 32'h0000ffdf;
        weights[0][31] = 32'h0000ffeb;
        weights[1][0] = 32'h0000ffef;
        weights[1][1] = 32'h0000ffe3;
        weights[1][2] = 32'h0000fff2;
        weights[1][3] = 32'h0000ffea;
        weights[1][4] = 32'h0000ffe6;
        weights[1][5] = 32'h0000ff7b;
        weights[1][6] = 32'h0000ffa7;
        weights[1][7] = 32'h0000fff9;
        weights[1][8] = 32'h00000073;
        weights[1][9] = 32'h0000ffdf;
        weights[1][10] = 32'h00000009;
        weights[1][11] = 32'h00000037;
        weights[1][12] = 32'h0000001f;
        weights[1][13] = 32'h0000002f;
        weights[1][14] = 32'h0000fff7;
        weights[1][15] = 32'h0000ffc5;
        weights[1][16] = 32'h00000002;
        weights[1][17] = 32'h00000002;
        weights[1][18] = 32'h0000fff9;
        weights[1][19] = 32'h0000000a;
        weights[1][20] = 32'h00000023;
        weights[1][21] = 32'h000000a1;
        weights[1][22] = 32'h00000083;
        weights[1][23] = 32'h0000002c;
        weights[1][24] = 32'h0000000f;
        weights[1][25] = 32'h00000021;
        weights[1][26] = 32'h0000fff1;
        weights[1][27] = 32'h00000017;
        weights[1][28] = 32'h0000ffec;
        weights[1][29] = 32'h00000052;
        weights[1][30] = 32'h0000005a;
        weights[1][31] = 32'h00000025;
        weights[2][0] = 32'h0000001d;
        weights[2][1] = 32'h0000fffe;
        weights[2][2] = 32'h0000000d;
        weights[2][3] = 32'h00000011;
        weights[2][4] = 32'h0000ffd4;
        weights[2][5] = 32'h00000012;
        weights[2][6] = 32'h00000044;
        weights[2][7] = 32'h0000ffc5;
        weights[2][8] = 32'h0000ffac;
        weights[2][9] = 32'h00000011;
        weights[2][10] = 32'h0000fff5;
        weights[2][11] = 32'h0000ffaa;
        weights[2][12] = 32'h0000fff2;
        weights[2][13] = 32'h0000ffe8;
        weights[2][14] = 32'h00000021;
        weights[2][15] = 32'h0000002a;
        weights[2][16] = 32'h0000ff9f;
        weights[2][17] = 32'h0000001b;
        weights[2][18] = 32'h00000000;
        weights[2][19] = 32'h0000ffb9;
        weights[2][20] = 32'h0000ff98;
        weights[2][21] = 32'h0000fffd;
        weights[2][22] = 32'h0000ffe0;
        weights[2][23] = 32'h0000ffbd;
        weights[2][24] = 32'h0000ffd3;
        weights[2][25] = 32'h00000012;
        weights[2][26] = 32'h00000006;
        weights[2][27] = 32'h0000ffd8;
        weights[2][28] = 32'h00000026;
        weights[2][29] = 32'h0000fff6;
        weights[2][30] = 32'h0000ffbb;
        weights[2][31] = 32'h0000ffe1;
        weights[3][0] = 32'h00000018;
        weights[3][1] = 32'h00000062;
        weights[3][2] = 32'h00000041;
        weights[3][3] = 32'h00000005;
        weights[3][4] = 32'h00000005;
        weights[3][5] = 32'h00000026;
        weights[3][6] = 32'h0000002b;
        weights[3][7] = 32'h00000017;
        weights[3][8] = 32'h0000ffe1;
        weights[3][9] = 32'h0000ffcc;
        weights[3][10] = 32'h0000ffc7;
        weights[3][11] = 32'h0000ffc2;
        weights[3][12] = 32'h00000010;
        weights[3][13] = 32'h0000ffe6;
        weights[3][14] = 32'h0000ffd5;
        weights[3][15] = 32'h00000003;
        weights[3][16] = 32'h00000013;
        weights[3][17] = 32'h0000ffe0;
        weights[3][18] = 32'h0000ffd7;
        weights[3][19] = 32'h00000025;
        weights[3][20] = 32'h0000ffb9;
        weights[3][21] = 32'h0000ff95;
        weights[3][22] = 32'h0000ffa0;
        weights[3][23] = 32'h0000ffca;
        weights[3][24] = 32'h00000015;
        weights[3][25] = 32'h0000ffc3;
        weights[3][26] = 32'h0000ffcd;
        weights[3][27] = 32'h0000fff8;
        weights[3][28] = 32'h0000002c;
        weights[3][29] = 32'h0000000a;
        weights[3][30] = 32'h0000000b;
        weights[3][31] = 32'h0000001f;
        weights[4][0] = 32'h0000004b;
        weights[4][1] = 32'h00000018;
        weights[4][2] = 32'h00000010;
        weights[4][3] = 32'h00000025;
        weights[4][4] = 32'h0000fffc;
        weights[4][5] = 32'h00000016;
        weights[4][6] = 32'h0000000e;
        weights[4][7] = 32'h0000ffd8;
        weights[4][8] = 32'h0000ffce;
        weights[4][9] = 32'h0000002e;
        weights[4][10] = 32'h0000001d;
        weights[4][11] = 32'h0000ffd8;
        weights[4][12] = 32'h0000ffe6;
        weights[4][13] = 32'h0000ffd8;
        weights[4][14] = 32'h0000000e;
        weights[4][15] = 32'h00000000;
        weights[4][16] = 32'h00000007;
        weights[4][17] = 32'h0000ffab;
        weights[4][18] = 32'h0000ffaa;
        weights[4][19] = 32'h00000000;
        weights[4][20] = 32'h0000ffc4;
        weights[4][21] = 32'h0000ffa3;
        weights[4][22] = 32'h0000ffcc;
        weights[4][23] = 32'h0000000f;
        weights[4][24] = 32'h0000ffca;
        weights[4][25] = 32'h0000ffd2;
        weights[4][26] = 32'h0000ffc4;
        weights[4][27] = 32'h0000ffd4;
        weights[4][28] = 32'h0000000c;
        weights[4][29] = 32'h0000ffcf;
        weights[4][30] = 32'h0000ffb2;
        weights[4][31] = 32'h0000ffb8;
        weights[5][0] = 32'h00000024;
        weights[5][1] = 32'h00000016;
        weights[5][2] = 32'h00000043;
        weights[5][3] = 32'h00000034;
        weights[5][4] = 32'h00000010;
        weights[5][5] = 32'h00000033;
        weights[5][6] = 32'h00000054;
        weights[5][7] = 32'h0000000d;
        weights[5][8] = 32'h0000ffda;
        weights[5][9] = 32'h0000ffba;
        weights[5][10] = 32'h0000ffb4;
        weights[5][11] = 32'h0000ffd0;
        weights[5][12] = 32'h0000fff7;
        weights[5][13] = 32'h0000fff9;
        weights[5][14] = 32'h0000fff5;
        weights[5][15] = 32'h00000008;
        weights[5][16] = 32'h0000ffab;
        weights[5][17] = 32'h0000005b;
        weights[5][18] = 32'h00000039;
        weights[5][19] = 32'h0000ffbc;
        weights[5][20] = 32'h0000ffa5;
        weights[5][21] = 32'h0000fff3;
        weights[5][22] = 32'h0000ffcc;
        weights[5][23] = 32'h0000ffb6;
        weights[5][24] = 32'h0000ffce;
        weights[5][25] = 32'h0000fffc;
        weights[5][26] = 32'h0000ffed;
        weights[5][27] = 32'h0000ffc9;
        weights[5][28] = 32'h00000022;
        weights[5][29] = 32'h0000ffe4;
        weights[5][30] = 32'h0000ffd6;
        weights[5][31] = 32'h0000ffdc;
        weights[6][0] = 32'h00000002;
        weights[6][1] = 32'h00000034;
        weights[6][2] = 32'h0000003b;
        weights[6][3] = 32'h0000001a;
        weights[6][4] = 32'h00000049;
        weights[6][5] = 32'h0000fffa;
        weights[6][6] = 32'h0000000b;
        weights[6][7] = 32'h00000059;
        weights[6][8] = 32'h00000026;
        weights[6][9] = 32'h00000037;
        weights[6][10] = 32'h0000ffe9;
        weights[6][11] = 32'h0000ffef;
        weights[6][12] = 32'h00000012;
        weights[6][13] = 32'h0000000a;
        weights[6][14] = 32'h00000016;
        weights[6][15] = 32'h0000ffd4;
        weights[6][16] = 32'h0000fff6;
        weights[6][17] = 32'h0000ffa4;
        weights[6][18] = 32'h0000ffbd;
        weights[6][19] = 32'h0000fff1;
        weights[6][20] = 32'h0000ffcf;
        weights[6][21] = 32'h0000ff45;
        weights[6][22] = 32'h0000ff12;
        weights[6][23] = 32'h0000ffc4;
        weights[6][24] = 32'h0000ffd5;
        weights[6][25] = 32'h0000ff6f;
        weights[6][26] = 32'h0000ffa2;
        weights[6][27] = 32'h0000fffb;
        weights[6][28] = 32'h0000ffd8;
        weights[6][29] = 32'h0000002c;
        weights[6][30] = 32'h0000003d;
        weights[6][31] = 32'h0000ffd4;
        weights[7][0] = 32'h0000ffe0;
        weights[7][1] = 32'h00000043;
        weights[7][2] = 32'h00000014;
        weights[7][3] = 32'h0000ffe1;
        weights[7][4] = 32'h0000001e;
        weights[7][5] = 32'h00000000;
        weights[7][6] = 32'h0000fff0;
        weights[7][7] = 32'h00000060;
        weights[7][8] = 32'h0000ffbc;
        weights[7][9] = 32'h00000021;
        weights[7][10] = 32'h0000005f;
        weights[7][11] = 32'h00000005;
        weights[7][12] = 32'h0000ffed;
        weights[7][13] = 32'h00000009;
        weights[7][14] = 32'h0000fffa;
        weights[7][15] = 32'h0000000c;
        weights[7][16] = 32'h00000021;
        weights[7][17] = 32'h00000002;
        weights[7][18] = 32'h0000003b;
        weights[7][19] = 32'h0000002f;
        weights[7][20] = 32'h0000001b;
        weights[7][21] = 32'h0000fffa;
        weights[7][22] = 32'h0000ffe8;
        weights[7][23] = 32'h00000015;
        weights[7][24] = 32'h0000ffb3;
        weights[7][25] = 32'h00000000;
        weights[7][26] = 32'h0000ffef;
        weights[7][27] = 32'h0000ff62;
        weights[7][28] = 32'h0000000b;
        weights[7][29] = 32'h0000ffa3;
        weights[7][30] = 32'h0000ff86;
        weights[7][31] = 32'h0000ff97;
        weights[8][0] = 32'h0000ffd8;
        weights[8][1] = 32'h0000ffaa;
        weights[8][2] = 32'h0000ffb1;
        weights[8][3] = 32'h0000ffa9;
        weights[8][4] = 32'h0000ffcd;
        weights[8][5] = 32'h0000ffbc;
        weights[8][6] = 32'h0000ffbb;
        weights[8][7] = 32'h0000ffb7;
        weights[8][8] = 32'h00000059;
        weights[8][9] = 32'h0000ffb4;
        weights[8][10] = 32'h0000ff9b;
        weights[8][11] = 32'h0000fff9;
        weights[8][12] = 32'h00000020;
        weights[8][13] = 32'h0000ffd9;
        weights[8][14] = 32'h0000ffe1;
        weights[8][15] = 32'h00000035;
        weights[8][16] = 32'h00000019;
        weights[8][17] = 32'h0000000d;
        weights[8][18] = 32'h0000ffed;
        weights[8][19] = 32'h0000001d;
        weights[8][20] = 32'h0000001f;
        weights[8][21] = 32'h0000003d;
        weights[8][22] = 32'h00000034;
        weights[8][23] = 32'h00000039;
        weights[8][24] = 32'h0000003f;
        weights[8][25] = 32'h0000001f;
        weights[8][26] = 32'h0000002e;
        weights[8][27] = 32'h00000058;
        weights[8][28] = 32'h0000002b;
        weights[8][29] = 32'h0000fffb;
        weights[8][30] = 32'h0000fffc;
        weights[8][31] = 32'h00000038;
        weights[9][0] = 32'h0000ffb1;
        weights[9][1] = 32'h0000fff6;
        weights[9][2] = 32'h0000fffd;
        weights[9][3] = 32'h0000ffa6;
        weights[9][4] = 32'h00000007;
        weights[9][5] = 32'h0000ffac;
        weights[9][6] = 32'h0000ffac;
        weights[9][7] = 32'h00000005;
        weights[9][8] = 32'h0000003e;
        weights[9][9] = 32'h0000fff4;
        weights[9][10] = 32'h00000016;
        weights[9][11] = 32'h00000031;
        weights[9][12] = 32'h00000032;
        weights[9][13] = 32'h0000001f;
        weights[9][14] = 32'h00000021;
        weights[9][15] = 32'h0000ffe6;
        weights[9][16] = 32'h00000020;
        weights[9][17] = 32'h00000021;
        weights[9][18] = 32'h0000002d;
        weights[9][19] = 32'h0000003a;
        weights[9][20] = 32'h00000013;
        weights[9][21] = 32'h0000002d;
        weights[9][22] = 32'h00000010;
        weights[9][23] = 32'h00000013;
        weights[9][24] = 32'h00000033;
        weights[9][25] = 32'h0000ffdf;
        weights[9][26] = 32'h0000ffde;
        weights[9][27] = 32'h00000024;
        weights[9][28] = 32'h00000025;
        weights[9][29] = 32'h00000021;
        weights[9][30] = 32'h00000023;
        weights[9][31] = 32'h00000045;
        biases[0] = 32'h0000012c;
        biases[1] = 32'h0000fdc0;
        biases[2] = 32'h0000019d;
        biases[3] = 32'h00000088;
        biases[4] = 32'h00000165;
        biases[5] = 32'h00000042;
        biases[6] = 32'h0000ff78;
        biases[7] = 32'h0000febf;
        biases[8] = 32'h0000018f;
        biases[9] = 32'h0000fdbb;
    end

    // Linear layer computation
    reg [31:0] linear_accumulator;
    reg [7:0] output_reg;
    reg [5:0] weight_counter;
    reg computation_done;

    always @(posedge clk) begin
        if (reset) begin
            linear_accumulator <= 32'b0;
            output_reg <= 8'd42; // Start with non-zero value
            weight_counter <= 6'b0;
            computation_done <= 1'b0;
        end else begin
            if (!computation_done) begin
                // Real linear layer using actual weights
                // For Tiny Tapeout, we'll compute a simplified dot product
                if (weight_counter < 6'd32) begin
                    linear_accumulator <= linear_accumulator + ({24'b0, input_data} * weights[0][weight_counter][15:0]);
                    weight_counter <= weight_counter + 1;
                end else begin
                    // Final computation with bias and saturation
                    if ((linear_accumulator >> 8) + biases[0][15:0] > 255) begin
                        output_reg <= 8'd255;
                    end else if ((linear_accumulator >> 8) + biases[0][15:0] < 0) begin
                        output_reg <= 8'd0;
                    end else begin
                        output_reg <= (linear_accumulator >> 8) + biases[0][15:0];
                    end
                    computation_done <= 1'b1;
                end
            end
        end
    end

    assign output_data = output_reg;

endmodule

// Simplified ReLU Layer for Tiny Tapeout
module relu_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data,
    output wire [7:0] output_data
);

    // Simplified ReLU for Tiny Tapeout
    reg [7:0] output_reg;

    always @(posedge clk) begin
        if (reset) begin
            output_reg <= 8'b0;
        end else begin
            // Simplified ReLU operation
            if (input_data[7] == 1'b0) begin
                output_reg <= input_data;
            end else begin
                output_reg <= 8'b0;
            end
        end
    end

    assign output_data = output_reg;

endmodule

// Simplified MaxPool Layer for Tiny Tapeout
module maxpool_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data,
    output wire [7:0] output_data
);

    // Simplified maxpool for Tiny Tapeout
    reg [7:0] output_reg;

    always @(posedge clk) begin
        if (reset) begin
            output_reg <= 8'b0;
        end else begin
            // Simplified maxpool operation
            output_reg <= input_data; // Pass through for simplicity
        end
    end

    assign output_data = output_reg;

endmodule
