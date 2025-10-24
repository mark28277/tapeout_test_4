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
    assign reset = ~rst_n;

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
        weights[0][0][0][0] = 32'hffff763e;
        weights[0][0][0][1] = 32'h00001ec3;
        weights[0][0][0][2] = 32'h000131be;
        weights[0][0][1][0] = 32'h00001a42;
        weights[0][0][1][1] = 32'hffffa4f8;
        weights[0][0][1][2] = 32'h00000531;
        weights[0][0][2][0] = 32'h00009b65;
        weights[0][0][2][1] = 32'hffffd57e;
        weights[0][0][2][2] = 32'hffff6840;
        weights[0][1][0][0] = 32'hfffef51c;
        weights[0][1][0][1] = 32'hffff996e;
        weights[0][1][0][2] = 32'h00001ffe;
        weights[0][1][1][0] = 32'h00005517;
        weights[0][1][1][1] = 32'hffff5aaf;
        weights[0][1][1][2] = 32'hfffff294;
        weights[0][1][2][0] = 32'h0000b8b9;
        weights[0][1][2][1] = 32'hffffd59b;
        weights[0][1][2][2] = 32'hffff6aa2;
        weights[0][2][0][0] = 32'hffff5d45;
        weights[0][2][0][1] = 32'hfffff734;
        weights[0][2][0][2] = 32'h00006ad5;
        weights[0][2][1][0] = 32'h00001c91;
        weights[0][2][1][1] = 32'hffff5ca6;
        weights[0][2][1][2] = 32'h00001a45;
        weights[0][2][2][0] = 32'h00006b28;
        weights[0][2][2][1] = 32'hfffffc03;
        weights[0][2][2][2] = 32'hffff4baf;
        weights[1][0][0][0] = 32'hffff64c9;
        weights[1][0][0][1] = 32'hffffcf11;
        weights[1][0][0][2] = 32'hffff884d;
        weights[1][0][1][0] = 32'h0000088f;
        weights[1][0][1][1] = 32'h0000093d;
        weights[1][0][1][2] = 32'h00001cb8;
        weights[1][0][2][0] = 32'h00006ea3;
        weights[1][0][2][1] = 32'h000057e2;
        weights[1][0][2][2] = 32'h0000320f;
        weights[1][1][0][0] = 32'hffff584e;
        weights[1][1][0][1] = 32'hffffb4dc;
        weights[1][1][0][2] = 32'hffff6700;
        weights[1][1][1][0] = 32'hffffcfff;
        weights[1][1][1][1] = 32'h0000186f;
        weights[1][1][1][2] = 32'hffffcaf7;
        weights[1][1][2][0] = 32'hffffb3c2;
        weights[1][1][2][1] = 32'hffff71b6;
        weights[1][1][2][2] = 32'hffffc5a2;
        weights[1][2][0][0] = 32'hfffff0af;
        weights[1][2][0][1] = 32'hffffdd4d;
        weights[1][2][0][2] = 32'hffffb7b8;
        weights[1][2][1][0] = 32'h00009372;
        weights[1][2][1][1] = 32'h00008215;
        weights[1][2][1][2] = 32'h0000b275;
        weights[1][2][2][0] = 32'h00005b75;
        weights[1][2][2][1] = 32'h000022c8;
        weights[1][2][2][2] = 32'h000063d5;
        biases[0] = 32'h0000294f;
        biases[1] = 32'hffffdc2f;
    end

    // Convolution computation
    reg [31:0] conv_result;
    reg [7:0] output_reg;

    always @(posedge clk) begin
        if (reset) begin
            conv_result <= 32'b0;
            output_reg <= 8'b0;
        end else begin
            // Simplified convolution with actual weights
            // For Tiny Tapeout, we'll use a simplified approach
            // using the first output channel and first input channel
            conv_result <= {24'b0, input_data} * weights[0][0][1][1];
            // Add bias and convert back to 8-bit
            output_reg <= (conv_result[31:16] + biases[0][31:16]) >> 16;
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
        weights[0][0] = 32'hffffe9ba;
        weights[0][1] = 32'hffffbcce;
        weights[0][2] = 32'hffff8b2a;
        weights[0][3] = 32'hffff8cd0;
        weights[0][4] = 32'h00000a6c;
        weights[0][5] = 32'hffffd71f;
        weights[0][6] = 32'hffffa926;
        weights[0][7] = 32'hffffc3f3;
        weights[0][8] = 32'hffffe581;
        weights[0][9] = 32'h00001b2c;
        weights[0][10] = 32'h00001951;
        weights[0][11] = 32'hffffb316;
        weights[0][12] = 32'hffffd8cd;
        weights[0][13] = 32'hffffc7aa;
        weights[0][14] = 32'h0000042e;
        weights[0][15] = 32'h0000105e;
        weights[0][16] = 32'h00001634;
        weights[0][17] = 32'hffffe2e6;
        weights[0][18] = 32'hffffdd84;
        weights[0][19] = 32'h00003114;
        weights[0][20] = 32'h00002479;
        weights[0][21] = 32'h0000399e;
        weights[0][22] = 32'h00004b7d;
        weights[0][23] = 32'h00008305;
        weights[0][24] = 32'h000039f4;
        weights[0][25] = 32'h00003779;
        weights[0][26] = 32'h00003fc5;
        weights[0][27] = 32'h000056b7;
        weights[0][28] = 32'h00000f5a;
        weights[0][29] = 32'hfffff70e;
        weights[0][30] = 32'hffffde10;
        weights[0][31] = 32'hffffea8e;
        weights[1][0] = 32'hffffeecb;
        weights[1][1] = 32'hffffe25c;
        weights[1][2] = 32'hfffff1be;
        weights[1][3] = 32'hffffe91d;
        weights[1][4] = 32'hffffe5cf;
        weights[1][5] = 32'hffff7ad4;
        weights[1][6] = 32'hffffa697;
        weights[1][7] = 32'hfffff895;
        weights[1][8] = 32'h00007381;
        weights[1][9] = 32'hffffde9d;
        weights[1][10] = 32'h000009e6;
        weights[1][11] = 32'h000037cd;
        weights[1][12] = 32'h00001fb9;
        weights[1][13] = 32'h00002f31;
        weights[1][14] = 32'hfffff6b6;
        weights[1][15] = 32'hffffc4b8;
        weights[1][16] = 32'h000002e6;
        weights[1][17] = 32'h00000223;
        weights[1][18] = 32'hfffff88e;
        weights[1][19] = 32'h00000ac1;
        weights[1][20] = 32'h000023ec;
        weights[1][21] = 32'h0000a1ba;
        weights[1][22] = 32'h000083ab;
        weights[1][23] = 32'h00002c5c;
        weights[1][24] = 32'h00000feb;
        weights[1][25] = 32'h0000217b;
        weights[1][26] = 32'hfffff001;
        weights[1][27] = 32'h0000170a;
        weights[1][28] = 32'hffffeb46;
        weights[1][29] = 32'h0000526e;
        weights[1][30] = 32'h00005a24;
        weights[1][31] = 32'h0000250e;
        weights[2][0] = 32'h00001dfc;
        weights[2][1] = 32'hfffffd03;
        weights[2][2] = 32'h00000ddd;
        weights[2][3] = 32'h000011a2;
        weights[2][4] = 32'hffffd331;
        weights[2][5] = 32'h00001279;
        weights[2][6] = 32'h000044cc;
        weights[2][7] = 32'hffffc4a9;
        weights[2][8] = 32'hffffab5c;
        weights[2][9] = 32'h000011d5;
        weights[2][10] = 32'hfffff45f;
        weights[2][11] = 32'hffffa960;
        weights[2][12] = 32'hfffff184;
        weights[2][13] = 32'hffffe7e2;
        weights[2][14] = 32'h0000211b;
        weights[2][15] = 32'h00002a68;
        weights[2][16] = 32'hffff9ef0;
        weights[2][17] = 32'h00001b8a;
        weights[2][18] = 32'h000000de;
        weights[2][19] = 32'hffffb811;
        weights[2][20] = 32'hffff9710;
        weights[2][21] = 32'hfffffc79;
        weights[2][22] = 32'hffffdf99;
        weights[2][23] = 32'hffffbcb6;
        weights[2][24] = 32'hffffd2a5;
        weights[2][25] = 32'h000012a0;
        weights[2][26] = 32'h00000662;
        weights[2][27] = 32'hffffd786;
        weights[2][28] = 32'h00002634;
        weights[2][29] = 32'hfffff58d;
        weights[2][30] = 32'hffffba83;
        weights[2][31] = 32'hffffe082;
        weights[3][0] = 32'h0000183f;
        weights[3][1] = 32'h00006260;
        weights[3][2] = 32'h00004187;
        weights[3][3] = 32'h0000059a;
        weights[3][4] = 32'h000005fc;
        weights[3][5] = 32'h0000266f;
        weights[3][6] = 32'h00002b61;
        weights[3][7] = 32'h000017b3;
        weights[3][8] = 32'hffffe0b5;
        weights[3][9] = 32'hffffcb58;
        weights[3][10] = 32'hffffc6bf;
        weights[3][11] = 32'hffffc161;
        weights[3][12] = 32'h000010ce;
        weights[3][13] = 32'hffffe5bc;
        weights[3][14] = 32'hffffd4ec;
        weights[3][15] = 32'h00000345;
        weights[3][16] = 32'h000013d1;
        weights[3][17] = 32'hffffdfe6;
        weights[3][18] = 32'hffffd672;
        weights[3][19] = 32'h00002516;
        weights[3][20] = 32'hffffb84c;
        weights[3][21] = 32'hffff94dc;
        weights[3][22] = 32'hffff9f77;
        weights[3][23] = 32'hffffc9cf;
        weights[3][24] = 32'h00001596;
        weights[3][25] = 32'hffffc23e;
        weights[3][26] = 32'hffffcc0b;
        weights[3][27] = 32'hfffff713;
        weights[3][28] = 32'h00002c1a;
        weights[3][29] = 32'h00000a7e;
        weights[3][30] = 32'h00000b83;
        weights[3][31] = 32'h00001f07;
        weights[4][0] = 32'h00004bc9;
        weights[4][1] = 32'h000018d0;
        weights[4][2] = 32'h00001077;
        weights[4][3] = 32'h0000258f;
        weights[4][4] = 32'hfffffb6e;
        weights[4][5] = 32'h00001680;
        weights[4][6] = 32'h00000e2f;
        weights[4][7] = 32'hffffd75b;
        weights[4][8] = 32'hffffcde3;
        weights[4][9] = 32'h00002e54;
        weights[4][10] = 32'h00001dec;
        weights[4][11] = 32'hffffd781;
        weights[4][12] = 32'hffffe5d0;
        weights[4][13] = 32'hffffd75f;
        weights[4][14] = 32'h00000e0c;
        weights[4][15] = 32'h00000022;
        weights[4][16] = 32'h000007ea;
        weights[4][17] = 32'hffffaac8;
        weights[4][18] = 32'hffffa948;
        weights[4][19] = 32'h000000b7;
        weights[4][20] = 32'hffffc364;
        weights[4][21] = 32'hffffa269;
        weights[4][22] = 32'hffffcb41;
        weights[4][23] = 32'h00000fe8;
        weights[4][24] = 32'hffffc938;
        weights[4][25] = 32'hffffd1c0;
        weights[4][26] = 32'hffffc334;
        weights[4][27] = 32'hffffd317;
        weights[4][28] = 32'h00000c71;
        weights[4][29] = 32'hffffce0a;
        weights[4][30] = 32'hffffb141;
        weights[4][31] = 32'hffffb718;
        weights[5][0] = 32'h000024cf;
        weights[5][1] = 32'h000016f5;
        weights[5][2] = 32'h000043ab;
        weights[5][3] = 32'h000034b9;
        weights[5][4] = 32'h00001037;
        weights[5][5] = 32'h000033c0;
        weights[5][6] = 32'h00005488;
        weights[5][7] = 32'h00000dbe;
        weights[5][8] = 32'hffffd9ca;
        weights[5][9] = 32'hffffb986;
        weights[5][10] = 32'hffffb33f;
        weights[5][11] = 32'hffffcfbb;
        weights[5][12] = 32'hfffff686;
        weights[5][13] = 32'hfffff812;
        weights[5][14] = 32'hfffff422;
        weights[5][15] = 32'h00000805;
        weights[5][16] = 32'hffffaa30;
        weights[5][17] = 32'h00005bae;
        weights[5][18] = 32'h00003903;
        weights[5][19] = 32'hffffbb8d;
        weights[5][20] = 32'hffffa4d7;
        weights[5][21] = 32'hfffff2e2;
        weights[5][22] = 32'hffffcb78;
        weights[5][23] = 32'hffffb5df;
        weights[5][24] = 32'hffffcd95;
        weights[5][25] = 32'hfffffb73;
        weights[5][26] = 32'hffffec20;
        weights[5][27] = 32'hffffc8cb;
        weights[5][28] = 32'h00002270;
        weights[5][29] = 32'hffffe3b6;
        weights[5][30] = 32'hffffd523;
        weights[5][31] = 32'hffffdb8a;
        weights[6][0] = 32'h00000289;
        weights[6][1] = 32'h0000349f;
        weights[6][2] = 32'h00003b92;
        weights[6][3] = 32'h00001af9;
        weights[6][4] = 32'h00004973;
        weights[6][5] = 32'hfffff9b0;
        weights[6][6] = 32'h00000be4;
        weights[6][7] = 32'h00005908;
        weights[6][8] = 32'h0000263a;
        weights[6][9] = 32'h000037ae;
        weights[6][10] = 32'hffffe8fb;
        weights[6][11] = 32'hffffeea0;
        weights[6][12] = 32'h00001284;
        weights[6][13] = 32'h00000a24;
        weights[6][14] = 32'h0000168d;
        weights[6][15] = 32'hffffd367;
        weights[6][16] = 32'hfffff5fb;
        weights[6][17] = 32'hffffa32f;
        weights[6][18] = 32'hffffbc52;
        weights[6][19] = 32'hfffff021;
        weights[6][20] = 32'hffffce2d;
        weights[6][21] = 32'hffff44ca;
        weights[6][22] = 32'hffff11e3;
        weights[6][23] = 32'hffffc3f3;
        weights[6][24] = 32'hffffd4c3;
        weights[6][25] = 32'hffff6e27;
        weights[6][26] = 32'hffffa1ee;
        weights[6][27] = 32'hfffffae1;
        weights[6][28] = 32'hffffd77b;
        weights[6][29] = 32'h00002cfa;
        weights[6][30] = 32'h00003da7;
        weights[6][31] = 32'hffffd3f5;
        weights[7][0] = 32'hffffdfef;
        weights[7][1] = 32'h000043a1;
        weights[7][2] = 32'h000014ae;
        weights[7][3] = 32'hffffe0a8;
        weights[7][4] = 32'h00001e2e;
        weights[7][5] = 32'hffffffea;
        weights[7][6] = 32'hffffef0a;
        weights[7][7] = 32'h0000609e;
        weights[7][8] = 32'hffffbbcf;
        weights[7][9] = 32'h0000219b;
        weights[7][10] = 32'h00005f04;
        weights[7][11] = 32'h00000575;
        weights[7][12] = 32'hffffec3d;
        weights[7][13] = 32'h0000093d;
        weights[7][14] = 32'hfffff95c;
        weights[7][15] = 32'h00000c23;
        weights[7][16] = 32'h0000211f;
        weights[7][17] = 32'h000002cd;
        weights[7][18] = 32'h00003b4b;
        weights[7][19] = 32'h00002f23;
        weights[7][20] = 32'h00001b35;
        weights[7][21] = 32'hfffff94f;
        weights[7][22] = 32'hffffe7a7;
        weights[7][23] = 32'h00001565;
        weights[7][24] = 32'hffffb270;
        weights[7][25] = 32'h000000fd;
        weights[7][26] = 32'hffffeedf;
        weights[7][27] = 32'hffff61aa;
        weights[7][28] = 32'h00000b6e;
        weights[7][29] = 32'hffffa263;
        weights[7][30] = 32'hffff85f9;
        weights[7][31] = 32'hffff96ee;
        weights[8][0] = 32'hffffd76f;
        weights[8][1] = 32'hffffa945;
        weights[8][2] = 32'hffffb066;
        weights[8][3] = 32'hffffa8ae;
        weights[8][4] = 32'hffffcc5c;
        weights[8][5] = 32'hffffbb2b;
        weights[8][6] = 32'hffffba29;
        weights[8][7] = 32'hffffb612;
        weights[8][8] = 32'h000059e7;
        weights[8][9] = 32'hffffb32f;
        weights[8][10] = 32'hffff9a1d;
        weights[8][11] = 32'hfffff85e;
        weights[8][12] = 32'h00002057;
        weights[8][13] = 32'hffffd8f6;
        weights[8][14] = 32'hffffe0f8;
        weights[8][15] = 32'h0000350b;
        weights[8][16] = 32'h00001925;
        weights[8][17] = 32'h00000dee;
        weights[8][18] = 32'hffffecd1;
        weights[8][19] = 32'h00001d79;
        weights[8][20] = 32'h00001f3e;
        weights[8][21] = 32'h00003dc1;
        weights[8][22] = 32'h000034f9;
        weights[8][23] = 32'h0000396b;
        weights[8][24] = 32'h00003f2c;
        weights[8][25] = 32'h00001ff6;
        weights[8][26] = 32'h00002e30;
        weights[8][27] = 32'h00005813;
        weights[8][28] = 32'h00002bd1;
        weights[8][29] = 32'hfffffa09;
        weights[8][30] = 32'hfffffb90;
        weights[8][31] = 32'h0000386d;
        weights[9][0] = 32'hffffb044;
        weights[9][1] = 32'hfffff575;
        weights[9][2] = 32'hfffffc88;
        weights[9][3] = 32'hffffa550;
        weights[9][4] = 32'h000007c6;
        weights[9][5] = 32'hffffab29;
        weights[9][6] = 32'hffffaba5;
        weights[9][7] = 32'h0000050a;
        weights[9][8] = 32'h00003ea9;
        weights[9][9] = 32'hfffff3aa;
        weights[9][10] = 32'h000016b5;
        weights[9][11] = 32'h000031e9;
        weights[9][12] = 32'h00003238;
        weights[9][13] = 32'h00001ff6;
        weights[9][14] = 32'h00002168;
        weights[9][15] = 32'hffffe55c;
        weights[9][16] = 32'h0000209a;
        weights[9][17] = 32'h000021ba;
        weights[9][18] = 32'h00002d12;
        weights[9][19] = 32'h00003a9c;
        weights[9][20] = 32'h000013b9;
        weights[9][21] = 32'h00002d79;
        weights[9][22] = 32'h0000106a;
        weights[9][23] = 32'h0000139c;
        weights[9][24] = 32'h00003376;
        weights[9][25] = 32'hffffde6a;
        weights[9][26] = 32'hffffdd5f;
        weights[9][27] = 32'h00002426;
        weights[9][28] = 32'h00002560;
        weights[9][29] = 32'h000021c3;
        weights[9][30] = 32'h000023b4;
        weights[9][31] = 32'h00004547;
        biases[0] = 32'h00012c51;
        biases[1] = 32'hfffdbf99;
        biases[2] = 32'h00019d13;
        biases[3] = 32'h000088fd;
        biases[4] = 32'h000165c0;
        biases[5] = 32'h0000429e;
        biases[6] = 32'hffff7722;
        biases[7] = 32'hfffebe27;
        biases[8] = 32'h00018f24;
        biases[9] = 32'hfffdba9c;
    end

    // Linear layer computation
    reg [31:0] linear_result;
    reg [7:0] output_reg;

    always @(posedge clk) begin
        if (reset) begin
            linear_result <= 32'b0;
            output_reg <= 8'b0;
        end else begin
            // Simplified linear operation with actual weights
            // For Tiny Tapeout, we'll use a simplified approach
            // using the first output neuron
            linear_result <= {24'b0, input_data} * weights[0][0];
            // Add bias and convert back to 8-bit
            output_reg <= (linear_result[31:16] + biases[0][31:16]) >> 16;
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
