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
    //-----------------------load image---------------------------
    reg [7:0] image_buffer [63:0];  // Store 64 pixels, based on input data
    reg [5:0] pixel_counter;      // Count 0-63
    wire loading_done = (pixel_counter == 63);

    always @(posedge clk) begin
        if (reset) begin
            pixel_counter <= 0;
            for (integer i = 0; i < 64; i = i + 1) begin
            image_buffer[i] <= 0;
            end
        end else begin
            image_buffer[pixel_counter] <= ui_in;  // Store current pixel
            if (pixel_counter < 63) begin
                pixel_counter <= pixel_counter + 1;
            end
        end
    end

    // Input interface for Tiny Tapeout limited I/O
    wire reset;
    assign reset = ~rst_n;
    
    // Conv2d Layer 0
    wire [7:0] conv_0_out_0;
    wire [7:0] conv_0_out_1; //based on # filters
    wire conv_0_valid;  // ADD THIS
    conv2d_layer conv_inst_0 (
        .clk(clk),
        .reset(reset),
        .input_data(image_buffer),
        .start_processing(loading_done),
        .output_data_0(conv_0_out_0),
        .output_data_1(conv_0_out_1),
        .output_valid(conv_0_valid)//for pipeline synchronization
    );

    // ReLU Layer 1
    wire [7:0] relu_1_out_0;
    wire [7:0] relu_1_out_1;
    wire relu_1_valid;
    relu_layer relu_inst_1 (
        .clk(clk),
        .reset(reset),
        .input_data_0(conv_0_out_0),
        .input_data_1(conv_0_out_1),
        .input_valid(conv_0_valid),
        .output_data_0(relu_1_out_0),
        .output_data_1(relu_1_out_1),
        .output_valid(relu_1_valid)
    );

    // MaxPool2d Layer 2
    wire [7:0] maxpool_2_out_0;
    wire [7:0] maxpool_2_out_1;
    wire maxpool_2_valid;
    maxpool_layer maxpool_inst_2 (
        .clk(clk),
        .reset(reset),
        .input_data_0(relu_1_out_0),
        .input_data_1(relu_1_out_1),
        .input_valid(relu_1_valid),
        .output_data_0(maxpool_2_out_0),
        .output_data_1(maxpool_2_out_1),
        .output_valid(maxpool_2_valid)
    );

    // Linear Layer 3
    wire [7:0] linear_3_out_0;
    wire [7:0] linear_3_out_1;
    wire linear_3_valid;
    linear_layer linear_inst_3 (
        .clk(clk),
        .reset(reset),
        .input_data_0(maxpool_2_out_0),
        .input_data_1(maxpool_2_out_1),
        .input_valid(maxpool_2_valid),
        .output_data_0(linear_3_out_0),
        .output_data_1(linear_3_out_1),
        .output_valid(linear_3_valid)
    );

    
    // Add these registers to collect ALL positions
    reg [7:0] feature_map_0 [35:0];  // Store all 36 positions for filter 0
    reg [7:0] feature_map_1 [35:0];  // Store all 36 positions for filter 1
    reg [5:0] collect_counter;       // Track which position we're collecting
    reg collection_done;             // Flag when all positions collected

    // Collection logic - runs after convolution completes
    always @(posedge clk) begin
        if (reset) begin
            collect_counter <= 0;
            collection_done <= 0;
        end else if (conv_0_valid) begin  // When conv outputs a valid position
            feature_map_0[collect_counter] <= conv_0_out_0;
            feature_map_1[collect_counter] <= conv_0_out_1;
            collect_counter <= collect_counter + 1;
            
            if (collect_counter == 35) begin
                collection_done <= 1;
                collect_counter <= 0;
            end
        end
    end

    // Simple prediction: average of all positions
    function [7:0] compute_prediction_0;
        integer i;
        reg [13:0] sum;  // 14-bit to avoid overflow (36*255 = 9180)
        begin
            sum = 0;
            for (i = 0; i < 36; i = i + 1) begin
                sum = sum + feature_map_0[i];
            end
            compute_prediction_0 = sum / 36;  // Average
        end
    endfunction

    function [7:0] compute_prediction_1;
        integer i;
        reg [13:0] sum;
        begin
            sum = 0;
            for (i = 0; i < 36; i = i + 1) begin
                sum = sum + feature_map_1[i];
            end
            compute_prediction_1 = sum / 36;  // Average
        end
    endfunction

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
            if (collection_done) begin
                // Output ACTUAL PREDICTION (average of all positions)
                uo_out_reg <= compute_prediction_0;   // Filter 0 average
                uio_out_reg <= compute_prediction_1;  // Filter 1 average
            end else begin
                // Still processing - show progress or zeros
                uo_out_reg <= 8'b0;
                uio_out_reg <= 8'b0;
            end
            uio_oe_reg <= 8'hFF;
        end
    end

    assign uo_out = uo_out_reg;
    assign uio_out = uio_out_reg;
    assign uio_oe = uio_oe_reg;

endmodule

//-------------conv2d module--------------------------------
module conv2d_layer (
    input wire clk,
    input wire reset,
    input wire start_processing,
    input wire [7:0] input_data [63:0],
    output reg [7:0] output_data_0, //output wire numbers based on # filters
    output reg [7:0] output_data_1,
    output reg output_valid
);

    // conv.weight
    // Shape: [2, 1, 3, 3] = 18 weights
    reg signed [7:0] conv_weight [17:0];
    always @(posedge clk) begin
        if(reset) begin
        conv_weight[0] <= 11;
        conv_weight[1] <= 8;
        conv_weight[2] <= 16;
        conv_weight[3] <= 9;
        conv_weight[4] <= 9;
        conv_weight[5] <= 14;
        conv_weight[6] <= -16;
        conv_weight[7] <= -12;
        conv_weight[8] <= 11;
        conv_weight[9] <= -11;
        conv_weight[10] <= -4;
        conv_weight[11] <= 4;
        conv_weight[12] <= -9;
        conv_weight[13] <= -16;
        conv_weight[14] <= 7;
        conv_weight[15] <= -7;
        conv_weight[16] <= -1;
        conv_weight[17] <= 10;
        end
    end

    // conv.bias
    // Shape: [2] = 2 weights
    reg signed [7:0] conv_bias [1:0];
    always @(posedge clk) begin
        if(reset) begin
        conv_bias[0] <= 3;
        conv_bias[1] <= 13;
        end
    end



    //-----------------convolution steps: ----------------------------
    reg processing;
    reg [5:0] pos_counter;    // 0-35
    reg [4:0] weight_counter; // 0-18

    // For window centered at (center_x, center_y), each pixel 0-8
    wire signed [2:0] center_x = pos_counter % 6;
    wire signed [2:0] center_y = pos_counter / 6;
    reg [7:0] input_buffer [8:0];

    function [7:0] get_pixel;
        input signed [4:0] x, y;  // Signed coordinates (-8 to 7)
        begin
            if (x < 0 || x > 7 || y < 0 || y > 7)
                get_pixel = 0;  // Zero-padding for edges
            else
                get_pixel = input_data[y * 8 + x];
        end
    endfunction

    always @(*) begin
        if (processing) begin
            input_buffer[0] = get_pixel((center_x-1), (center_y-1)); // Top-left
            input_buffer[1] = get_pixel(center_x, (center_y-1));     // Top-middle
            input_buffer[2] = get_pixel((center_x+1), (center_y-1)); // Top-right
            input_buffer[3] = get_pixel((center_x-1), center_y);     // Middle-left  
            input_buffer[4] = get_pixel(center_x, center_y);         // Center
            input_buffer[5] = get_pixel((center_x+1), center_y);     // Middle-right
            input_buffer[6] = get_pixel((center_x-1), (center_y+1)); // Bottom-left
            input_buffer[7] = get_pixel(center_x, (center_y+1));     // Bottom-middle
            input_buffer[8] = get_pixel((center_x+1), (center_y+1)); // Bottom-right
        end
    end
    
    
    //------------convolution: output of each convolution = Σ(pixel*weight)+bias-------------------
    reg [18:0] accum_0;  // Filter 0
    reg [18:0] accum_1;  // Filter 1
    reg [3:0] kernel_position;
    reg [7:0] pixel_val;

    // Scaling + ReLU function
    function [7:0] scale_and_relu;
        input [18:0] value;
        begin
            if (value[18]) begin
                // Negative → ReLU to 0
                scale_and_relu = 8'b0;
            end else if (value[18:11] != 8'b0) begin
                // Overflow → saturate to 255
                scale_and_relu = 8'hFF;
            end else begin
                // Normal case: scale 19-bit → 8-bit
                scale_and_relu = value[10:3];
            end
        end
    endfunction

    always @(posedge clk) 
    begin
        if (reset) begin
            // Reset everything
            weight_counter <= 0;
            accum_0 <= 0;
            accum_1 <= 0;
            output_data_0 <= 0;
            output_data_1 <= 0;
            output_valid <= 0;
            pos_counter <= 0;
            processing <= 0;
        end else if (start_processing && !processing) begin
            // Start processing after image loaded
            processing <= 1;
        end else if (processing) begin
            // INNER LOOP: Process weights for current position
            kernel_position = weight_counter % 9;
            pixel_val = input_buffer[kernel_position];
            if(weight_counter < 9) begin                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                accum_0 <= accum_0 + (pixel_val * conv_weight[weight_counter]);
            end else begin
                accum_1 <= accum_1 + (pixel_val * conv_weight[weight_counter]);
            end

            if (weight_counter == 17) //based on # weights
            begin
                output_data_0 <= scale_and_relu(accum_0 + (conv_bias[0] << 11));
                output_data_1 <= scale_and_relu(accum_1 + (conv_bias[1] << 11));
                output_valid <= 1;
                weight_counter <= 0;
                accum_0 <= 0;  // Reset accumulators for next position
                accum_1 <= 0;
                pos_counter <= pos_counter + 1; //move to nest position
            end else begin
                weight_counter <= weight_counter + 1;
                output_valid <= 0;
            end

            if (pos_counter == 35) begin
                processing <= 0;  // All done!
            end
        end else begin
            output_valid <= 0;
        end
    end
endmodule
//--------------------------------------------------------------------------------------------------------------

//---------------Simplified Linear Layer for Tiny Tapeout----------------------------
module linear_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data_0,
    input wire [7:0] input_data_1,
    input wire input_valid,
    output wire [7:0] output_data_0,
    output wire [7:0] output_data_1,
    output reg output_valid
);

    //Simplified linear layer for Tiny Tapeout
    reg [7:0] output_reg_0;
    reg [7:0] output_reg_1;

    always @(posedge clk) begin
        if (reset) begin
            output_reg_0 <= 8'b0;
            output_reg_1 <= 8'b0;
            output_valid <= 0;
        end else if (input_valid) begin
            // Simplified linear operation
            output_reg_0 <= input_data_0 + 8'h20;
            output_reg_1 <= input_data_1 + 8'h20;
            output_valid <= 1;
        end else begin
            output_valid <= 0;
        end
    end

    assign output_data_0 = output_reg_0;
    assign output_data_1 = output_reg_1;

endmodule

// -----------Simplified ReLU Layer for Tiny Tapeout---------------------
module relu_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data_0,
    input wire [7:0] input_data_1,
    input wire input_valid,
    output wire [7:0] output_data_0,
    output wire [7:0] output_data_1,
    output reg output_valid
    
);

    // Simplified ReLU for Tiny Tapeout
    reg [7:0] output_reg_0;
    reg [7:0] output_reg_1;

    always @(posedge clk) begin
        if (reset) begin
            output_reg_0 <= 8'b0;
            output_reg_1 <= 8'b0;
            output_valid <= 0;
        end else if (input_valid) begin
            // Simplified ReLU operation
            output_reg_0 <= (input_data_0[7] == 1'b0) ? input_data_0 : 8'b0;
            output_reg_1 <= (input_data_1[7] == 1'b0) ? input_data_1 : 8'b0;
            output_valid <= 1;
        end else begin
            output_valid <= 0;
        end
    end

    assign output_data_0 = output_reg_0;
    assign output_data_1 = output_reg_1;

endmodule

// -----------Simplified MaxPool Layer for Tiny Tapeout-------------------------
module maxpool_layer (
    input wire clk,
    input wire reset,
    input wire [7:0] input_data_0,
    input wire [7:0] input_data_1,
    input wire input_valid,
    output wire [7:0] output_data_0,
    output wire [7:0] output_data_1,
    output reg output_valid
);

    // Simplified maxpool for Tiny Tapeout
    reg [7:0] output_reg_0;
    reg [7:0] output_reg_1;

    always @(posedge clk) begin
        if (reset) begin
            output_reg_0 <= 8'b0;
            output_reg_1 <= 8'b0;
            output_valid <= 0;
        end else if (input_valid) begin
            // Simplified maxpool operation
            output_reg_0 <= input_data_0; // Pass through for simplicity
            output_reg_1 <= input_data_1;
            output_valid <= 1;
        end else begin
            output_valid <= 0;
        end
    end

    assign output_data_0 = output_reg_0;
    assign output_data_1 = output_reg_1;

endmodule
