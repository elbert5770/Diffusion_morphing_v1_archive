using DataFrames, CSV, JLD2
using SparseArrays
using Images

function read_and_process_sframe_files(csv_path1::String)
    # Read CSV files without headers
    data1 = try
        df1 = DataFrame(CSV.File(csv_path1, header=false))
        df1[2:end, :]  # Remove first row
    catch e
        println("Error reading first CSV file: ", e)
        return nothing
    end
    
    return data1
end

function record_sparse_arrays(data)
    nx,ny = size(data)

    # Get unique values from all datasets
    IDs = unique(vec(Matrix(data)))
    
    
    # Create sparse arrays for each value in each dataset
    sparse_arrays = Dict{Tuple{Int64,Int64}, SparseMatrixCSC{Float64, Int64}}()
    sparse_arrays_components = Dict{Int64,Int64}()
    for ID in IDs
        mask_binary = Matrix(data) .== ID

        components = Images.label_components(mask_binary)
        @show ID,maximum(components)
        for i in 1:maximum(components)
            mask_component = components .== i
            if any(mask_component)
                indices = findall(mask_component)
                rows = [idx[1] for idx in indices]
                cols = [idx[2] for idx in indices]
                vals = Float64.(fill(1, length(rows)))
                
                # Create sparse array with same dimensions as original data
                sparse_arrays[(ID,i)] = sparse(rows, cols, vals, nx, ny)
                
            end
        end
        sparse_arrays_components[ID] = maximum(components)
    end
    
    
    return sparse_arrays,sparse_arrays_components
end

function main(path::String,slice::Int64,xloc::Int64,yloc::Int64,height::Int64)
    # Define file paths
    
    
    csv_file = string(path,"sframe_$(slice)_$(xloc)_$(yloc)_$(height).csv")
    # Read the files
    data1 = read_and_process_sframe_files(csv_file)
    # @show size(data1)
    
  
    
    # Plot the points and get sparse arrays
    sparse_arrays,sparse_arrays_components = record_sparse_arrays(data1)


    
    # Save sparse arrays to JLD2 file
    jld2_path = string(path,"sparse_arrays_",slice,"_",xloc,"_",yloc,"_",height,".jld2")
    @save jld2_path sparse_arrays sparse_arrays_components
    println("Sparse arrays saved to: ", jld2_path)
end


sim_number = parse(Int64,ARGS[1])
path = string(@__DIR__, "/../Neuron_transport_data/")
filename = string(@__DIR__,"/Simulation_settings_4.0.csv")
settings = CSV.File(filename) |> Tables.matrix

        
xloc::Int64 = settings[sim_number,2]
yloc::Int64 = settings[sim_number,3]
start_slice::Int64 = settings[sim_number,4]
height::Int64 = settings[sim_number,5]




slices = Int64(height*25)




for slice in start_slice:start_slice+slices
    println("Processing slice: $slice")
    main(path, slice, xloc, yloc, height)
end
