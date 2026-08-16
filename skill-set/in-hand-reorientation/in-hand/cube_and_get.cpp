#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono> 
#include <filesystem>
#include <ctime>
#include <sstream>
#include <thread>
#include <iomanip>
#include <H5Cpp.h>
#include <string>
#include <stdexcept>
#include <charconv>
#include <array>  

// 2026.01.13.
// manual cube rotate

// python trajectory_utils/smooth.py --in ./data/collected_data_0129_1648.hdf5 --out ./data/collected_data_0129_1648.hdf5 
#define H5FILE_NAME "./data/collected_data_0129_1648.hdf5"
#define DATASET_PATH "/data/demo_2/sim_joint_real_smooth"
// #define DATASET_PATH "/data/demo_0/sim_joint_real_smooth"
// #define DATASET_PATH "/data/demo_1/sim_joint_real"

#define SCALE_FACTOR 4096
// #define SCALE_FACTOR2 2.16
// #define SCALE_FACTOR2 2.125
#define SCALE_FACTOR2 2.145
#define SCALE_FACTOR3 1.1
// #define SCALE_RING 1.
// #define SCALE_IM 1.1

#define PI 3.14159265358979323846
#define UDP_PORT 12345
#define SERVER_IP "192.168.1.250"
// #define FRAME_INTERVAL_MS 15 // 1/100초
#define FRAME_INTERVAL_MS 15 // 1/100초
using namespace H5;

int main() {
    try {
        // HDF5 파일 열기
        std::cout << "Opening HDF5 file..." << std::endl;
        H5::H5File file(H5FILE_NAME, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(DATASET_PATH);

        H5::DataSpace dataspace = dataset.getSpace();
 
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, NULL);
        std::cout << "Dataset dimensions: " << dims[0] << "x" << dims[1] << std::endl;

        if (dims[0] == 0 || dims[1] == 0) {
            std::cerr << "Error: Dataset is empty." << std::endl;
            return 1;
        }

        std::vector<float> raw_data(dims[0] * dims[1]);
        std::vector<int16_t> processed_data(dims[0] * dims[1]);


        dataset.read(raw_data.data(), H5::PredType::NATIVE_FLOAT);

        // //ring
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 12;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= SCALE_RING * 1.1;  // 해당 값만 변경
        //     // raw_data[index] *= 0.4;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {   // 새끼 구부리기
        //     size_t index = i * dims[1] + 13;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     // raw_data[index] *= 0.85;  // 해당 값만 변경
        //     raw_data[index] *= SCALE_RING * 1.2;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 14;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     // raw_data[index] *= 0.9;  // 해당 값만 변경
        //     raw_data[index] *= SCALE_RING ;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 15;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 1.0;  // 해당 값만 변경
        //     // raw_data[index] *= SCALE_RING;  // 해당 값만 변경
        // }

        
        // //middle
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 8;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 1.1 ;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 9;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     // raw_data[index] *= 1.04;  // 해당 값만 변경
        //     raw_data[index] *= 1.12;  // 해당 값만 변경
        //     // raw_data[index] *= SCALE_IM;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 10;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     // raw_data[index] *= 0.9;  // 해당 값만 변경
        //     raw_data[index] *= 1.13;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 11;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 1.03;  // 해당 값만 변경
        // }

        // //index
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 4;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= SCALE_IM;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 6;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= SCALE_IM;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 7;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= SCALE_IM;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 8;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= SCALE_IM;  // 해당 값만 변경
        // }

        // thumb
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 1;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 0.5;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 1;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 0.5;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 2;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 0.75;  // 해당 값만 변경
        // }
        // for (size_t i = 0; i < dims[0]; ++i) {  
        //     size_t index = i * dims[1] + 3;  // i번째 행의 5번째 조인트 값 (index 5)  
        //     raw_data[index] *= 0.85;  // 해당 값만 변경
        // }

        for (size_t i = 0; i < raw_data.size(); ++i) {
            // processed_data[i] = static_cast<int16_t>((raw_data[i] * SCALE_FACTOR * SCALE_FACTOR2 * SCALE_FACTOR3) / PI);
            processed_data[i] = static_cast<int16_t>((raw_data[i] * SCALE_FACTOR * SCALE_FACTOR2 * SCALE_FACTOR3 ) / PI);
        }

        // UDP 소켓 설정
        int sockfd_send = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_send < 0) {
            std::cerr << "Error creating send socket" << std::endl;
            return 1;
        }

        struct sockaddr_in server_addr, recv_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(UDP_PORT);
        inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

        memset(&recv_addr, 0, sizeof(recv_addr));
        recv_addr.sin_family = AF_INET;
        recv_addr.sin_port = htons(UDP_PORT);
        recv_addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sockfd_send, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) < 0) {
            std::cerr << "Error binding receive socket" << std::endl;
            close(sockfd_send);
            return 1;
        }
        std::cout << "Sending data sequentially..." << std::endl;
        char buffer[1024];
        char buffer_get[2048];
        socklen_t client_len = sizeof(recv_addr);
        std::vector<std::vector<float>> received_data;
        std::vector<std::array<double, 2>> timestamps;

        for (size_t i = 0; i < dims[0]; ++i) {
            size_t offset = i * dims[1];
            std::string message;

            for (size_t j = 0; j < dims[1]; ++j) {
                message += std::to_string(processed_data[offset + j]);
                message += ",";
            }
            // 뒤에 int 1,1 추가
            message += "1,1\n";

            strncpy(buffer, message.c_str(), sizeof(buffer));
            buffer[sizeof(buffer) - 1] = '\0';

            // send time point
            // ==============================================================================
            auto send_time_point = std::chrono::high_resolution_clock::now();
            // ==============================================================================

            ssize_t sent = sendto(sockfd_send, buffer, strlen(buffer), 0,
                                  (struct sockaddr*)&server_addr, sizeof(server_addr));
            if (sent < 0) {
                std::cerr << "Error sending data" << std::endl;
                close(sockfd_send);
                return 1;
            }
            std::cout << "Sent row " << i + 1 << " of " << dims[0] << std::endl;
            std::cout << "Sent: " << message << std::endl; // For debugging

            ssize_t received = recvfrom(sockfd_send, buffer_get, sizeof(buffer_get) - 1, 0,
                                        (struct sockaddr*)&recv_addr, &client_len);

            // receive time point         
            // ==============================================================================
            auto receive_time_point = std::chrono::high_resolution_clock::now();
            // ==============================================================================
            
            if (received < 0) {
                std::cerr << "Error receiving data" << std::endl;
                break;
            }

            buffer_get[received] = '\0';
            std::cout << "Received: " << buffer_get << std::endl; // For debugging

            std::vector<float> row_data;
            std::istringstream in(buffer_get);
            std::string token;
            while (std::getline(in, token, ',')) {
                try {
                    float v = std::stof(token);
                    row_data.push_back(v);
                } catch (...) {
                    std::cerr << "Bad token: " << token << "\n";
                }
            }

            if (!row_data.empty()) {
                received_data.push_back(row_data);
                // Save timestamps
                double send_time = std::chrono::duration<double>(send_time_point.time_since_epoch()).count();
                double receive_time = std::chrono::duration<double>(receive_time_point.time_since_epoch()).count();
                timestamps.push_back({send_time, receive_time});
                // ======================================================================
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(FRAME_INTERVAL_MS));
        }

        close(sockfd_send);

        // HDF5 file saving
        std::time_t t = std::time(nullptr);
        std::tm* now = std::localtime(&t);
        std::ostringstream folder_ss;
        folder_ss << "./"
                << std::setw(2) << std::setfill('0') << (now->tm_mon + 1) // month
                << std::setw(2) << std::setfill('0') << now->tm_mday;   // day
        std::string folder = folder_ss.str();
        if (!std::filesystem::exists(folder)) {
            std::filesystem::create_directories(folder);
        }

        std::ostringstream oss;
        oss << folder << "/" << "cube_"
            << now->tm_hour << now->tm_min << now->tm_sec << ".hdf5";
        std::string save_name = oss.str();

        H5::H5File file2(save_name, H5F_ACC_TRUNC);

        if (received_data.empty()) {
             std::cout << "No data received, skipping file save." << std::endl;
             return 0;
        }

        hsize_t N = received_data.size();
        if (N != timestamps.size()) {
             std::cerr << "Error: Mismatch between received data count and timestamp count." << std::endl;
             return 1;
        }

        hsize_t row_size = received_data[0].size();
        if (row_size < 111) {
            std::cerr << "Error: Received data row size is " << row_size << ", which is less than the expected 111." << std::endl;
            return 1;
        }

        // Define dimensions for datasets
        hsize_t dims_joint_angle[2]   = { N, 16 };
        hsize_t dims_joint_current[2] = { N, 16 };
        hsize_t dims_ft[2]            = { N, 12 };
        hsize_t dims_contact[2]       = { N, 60 };
        hsize_t dims_time[2]          = { N, 2 };
        hsize_t obj_pose[7]          = { N, 7 };
        hsize_t tip_pos[12]          = { N, 12 };
        hsize_t tip_quat[16]          = { N, 16 };
        // ==============================================================================

        // Create dataspaces
        DataSpace space_joint_angle(2, dims_joint_angle);
        DataSpace space_joint_current(2, dims_joint_current);
        DataSpace space_ft(2, dims_ft);
        DataSpace space_contact(2, dims_contact);
        DataSpace space_time(2, dims_time);
        DataSpace space_obj_pose(2, obj_pose);
        DataSpace space_tip_pos(2, tip_pos);
        DataSpace space_tip_quat(2, tip_quat);
        // ===============================================================================

        // Create datasets
        DataSet dset_joint_angle   = file2.createDataSet("/joint_angle",   PredType::NATIVE_FLOAT, space_joint_angle);
        DataSet dset_joint_current = file2.createDataSet("/joint_current", PredType::NATIVE_FLOAT, space_joint_current);
        DataSet dset_ft            = file2.createDataSet("/FT",            PredType::NATIVE_FLOAT, space_ft);
        DataSet dset_contact       = file2.createDataSet("/contact",       PredType::NATIVE_FLOAT, space_contact);
        DataSet dset_time          = file2.createDataSet("/time",          PredType::NATIVE_DOUBLE, space_time);
        DataSet dset_obj_pose         = file2.createDataSet("/obj_pose",          PredType::NATIVE_FLOAT, space_obj_pose);
        DataSet dset_tip_pos         = file2.createDataSet("/tip_pos",          PredType::NATIVE_FLOAT, space_tip_pos);
        DataSet dset_tip_quat         = file2.createDataSet("/tip_quat",          PredType::NATIVE_FLOAT, space_tip_quat);

        // ===============================================================================

        // Prepare data buffers
        std::vector<float> buf_joint_angle(N * 16);
        std::vector<float> buf_joint_current(N * 16);
        std::vector<float> buf_ft(N * 12);
        std::vector<float> buf_contact(N * 60);
        std::vector<float> buf_obj_pose(N * 7);
        std::vector<float> buf_tip_pos(N * 12);
        std::vector<float> buf_tip_quat(N * 16);

        for (hsize_t i = 0; i < N; ++i) {
            const auto& row = received_data[i];
            std::copy(row.begin() + 0, row.begin() + 16, buf_joint_angle.begin() + i*16);
            std::copy(row.begin() + 16, row.begin() + 32, buf_joint_current.begin() + i*16);
            std::copy(row.begin() + 32, row.begin() + 44, buf_ft.begin() + i*12);
            std::copy(row.begin() + 44, row.begin() + 104, buf_contact.begin() + i*60);
            std::copy(row.begin() + 104, row.begin() + 111, buf_obj_pose.begin() + i*7);
            std::copy(row.begin() + 111, row.begin() + 123, buf_tip_pos.begin() + i*12);
            std::copy(row.begin() + 123, row.begin() + 139, buf_tip_quat.begin() + i*16);

        }

        std::vector<double> time_data_flat;
        time_data_flat.reserve(N * 2);
        for (const auto& ts_pair : timestamps) {
            time_data_flat.push_back(ts_pair[0]); // Send time
            time_data_flat.push_back(ts_pair[1]); // Receive time
        }

        // Write data to datasets
        dset_joint_angle.write(buf_joint_angle.data(), PredType::NATIVE_FLOAT);
        dset_joint_current.write(buf_joint_current.data(), PredType::NATIVE_FLOAT);
        dset_ft.write(buf_ft.data(), PredType::NATIVE_FLOAT);
        dset_contact.write(buf_contact.data(), PredType::NATIVE_FLOAT);
        dset_time.write(time_data_flat.data(), PredType::NATIVE_DOUBLE);
        dset_obj_pose.write(buf_obj_pose.data(), PredType::NATIVE_FLOAT);
        dset_tip_pos.write(buf_tip_pos.data(), PredType::NATIVE_FLOAT);
        dset_tip_quat.write(buf_tip_quat.data(), PredType::NATIVE_FLOAT);

        // ===================================================================================

        std::cout << "Data and timestamps saved successfully to " << save_name << std::endl;

    } catch (const H5::Exception& error) {
        error.printErrorStack();
        return 1;
    }

    return 0;
}
