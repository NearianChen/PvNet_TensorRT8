
#include "ransac_voting.h"
#include "ransac_helper.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
struct RansacState {
    std::vector<cv::Point2f> best_hypothesis;   // 最佳假设点
    std::vector<int> best_inlier_counts;       // 内点计数
    std::vector<std::vector<bool>> best_inliers; // 内点分布
    std::vector<cv::Point2f> previous_output_points; // 上一次的输出点
    bool is_initialized = false;               // 是否为首次运行
};

cv::Mat invertMatrix(const cv::Mat& matrix) {
    cv::Mat invMatrix;
    if (cv::invert(matrix, invMatrix, cv::DECOMP_SVD)) {
        return invMatrix;
    } else {
        throw std::runtime_error("Matrix inversion failed");
    }
}

cv::Point2f refinePointUsingInliersV2(
    const std::vector<cv::Point2f>& coords,
    const std::vector<cv::Vec2f>& directions,
    const std::vector<bool>& inliers
) {
    size_t tn = coords.size();
    cv::Matx22d ATA(0, 0, 0, 0);
    cv::Vec2d ATb(0, 0);

    for (size_t i = 0; i < tn; ++i) {
        if (!inliers[i]) continue;

        cv::Vec2f normal(-directions[i][1], directions[i][0]); // 法向量
        cv::Point2f coord = coords[i];

        // 更新 ATA
        ATA(0, 0) += normal[0] * normal[0];
        ATA(0, 1) += normal[0] * normal[1];
        ATA(1, 0) += normal[1] * normal[0];
        ATA(1, 1) += normal[1] * normal[1];

        // 更新 ATb
        double b = normal[0] * coord.x + normal[1] * coord.y;
        ATb[0] += normal[0] * b;
        ATb[1] += normal[1] * b;
    }

    // 求解 ATA * x = ATb
    cv::Matx22d ATA_inv;
    if (cv::invert(ATA, ATA_inv)) {
        cv::Vec2d refined_point = ATA_inv * ATb;
        return cv::Point2f(refined_point[0], refined_point[1]);
    } else {
        return cv::Point2f(0, 0); // 返回默认值以防失败
    }
}

void filterForegroundPoints(
    std::vector<cv::Point2f>& coords,
    int max_num,
    std::mt19937& rng
) {
    if (coords.size() > max_num) {
        // 直接随机打乱并截断，确保采样数量严格等于 max_num
        std::shuffle(coords.begin(), coords.end(), rng);
        coords.resize(max_num);
    }
}

namespace ransacVoting{

    void ransacVotingCUDAV2(
        cv::Mat& mask, std::vector<cv::Mat>& vertex, 
        std::vector<cv::Point2f>& output_points, 
        int round_hyp_num, float inlier_thresh, float confidence, 
        int max_iter, int min_num, int max_num
    ) {
        std::mt19937 rng(0);
        // std::random_device rd;  
        // std::mt19937 rng(rd()); 
        std::vector<cv::Point2f> coords;
        cv::findNonZero(mask, coords);
        int tn = coords.size();
        int vn = vertex.size();
        output_points.resize(vn, cv::Point2f(0, 0));

        if (tn < min_num) {
            std::cerr << "Not enough foreground points!" << std::endl;
            return;
        }
        if (tn > max_num) {
            filterForegroundPoints(coords, max_num, rng);
            tn = coords.size();
            // std::cout<< "tn: "<<tn<<std::endl;
        }

        const int hn = round_hyp_num;
        const float inlier_thresh_sq = inlier_thresh * inlier_thresh;

        // 预转换coords数据
        std::vector<float> h_coords(tn * 2);
        for (int i = 0; i < tn; ++i) {
            h_coords[i*2] = coords[i].x;
            h_coords[i*2+1] = coords[i].y;
        }

        for (int k = 0; k < vn; ++k) {
            // ============== 数据准备 ==============
            cv::Mat& vertex_k = vertex[k];
            const int h = vertex_k.rows;
            const int w = vertex_k.cols;

            // 转换vertex数据
            std::vector<float> h_direct(h * w * 2);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    cv::Vec2f offset = vertex_k.at<cv::Vec2f>(y, x);
                    h_direct[(y*w+x)*2] = offset[0];
                    h_direct[(y*w+x)*2+1] = offset[1];
                }
            }
            // std::cout<<"done1"<<std::endl;
            // ============== GPU内存分配 ==============
            float *d_direct, *d_coords, *d_hypo_pts;
            int *d_idxs;
            cudaMalloc(&d_direct, h*w*2*sizeof(float));
            // cudaMalloc(&d_coords, tn*2*sizeof(float));
            cudaMalloc(&d_idxs, hn*2*sizeof(int));
            cudaMalloc(&d_hypo_pts, hn*2*sizeof(float));
            cudaError_t err;
            err = cudaMalloc(&d_coords, tn*2*sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc d_direct failed: " 
                        << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
            // std::cout<<"done2"<<std::endl;
            // 初始化固定数据
            cudaMemcpy(d_direct, h_direct.data(), h*w*2*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_coords, h_coords.data(), tn*2*sizeof(float), cudaMemcpyHostToDevice);

            // ============== 投票相关内存 ==============
            int* d_inlier_counts;
            bool* d_inliers;
            cudaMalloc(&d_inlier_counts, hn*sizeof(int));
            cudaMalloc(&d_inliers, hn*tn*sizeof(bool));

            // ============== RANSAC主循环 ==============
            float best_ratio = 0.0f;
            cv::Point2f best_point(0, 0);
            int hyp_num = 0;
            std::vector<bool> best_inlier_flags(tn, false);

            for (int iter = 0; iter < max_iter; ++iter) {
                // 1. 生成随机索引
                std::vector<int> h_idxs(hn*2);
                std::uniform_int_distribution<int> dist(0, tn-1);
                for (int r = 0; r < hn*2; ++r) {
                    h_idxs[r] = dist(rng);
                }
                // std::cout << "Launching hypothesis generation with parameters:"
                //         << " h=" << h << " w=" << w
                //         << " tn=" << tn << " hn=" << hn << std::endl;            
                cudaMemcpy(d_idxs, h_idxs.data(), hn*2*sizeof(int), cudaMemcpyHostToDevice);

                // 2. 生成假设点
                generate_hypothesis_cuda(d_direct, d_coords, d_idxs, d_hypo_pts, h, w, tn, hn, 0);

                // 3. 执行投票
                cudaMemset(d_inlier_counts, 0, hn*sizeof(int));
                cudaMemset(d_inliers, 0, hn*tn*sizeof(bool));
                launch_voting_cuda(d_coords, d_direct, d_hypo_pts, d_inlier_counts, d_inliers,
                                h, w, tn, hn, inlier_thresh);

                // 4. 获取结果
                std::vector<int> h_inlier_counts(hn);
                std::vector<unsigned char> h_inliers(hn*tn);
                cudaMemcpy(h_inlier_counts.data(), d_inlier_counts, hn*sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_inliers.data(), d_inliers, hn*tn*sizeof(bool), cudaMemcpyDeviceToHost);

                // 5. 找到当前迭代的最佳假设
                auto max_it = std::max_element(h_inlier_counts.begin(), h_inlier_counts.end());
                int cur_max_idx = std::distance(h_inlier_counts.begin(), max_it);
                float cur_ratio = static_cast<float>(*max_it) / tn;

                // 6. 更新全局最佳结果
                if (cur_ratio > best_ratio) {
                    best_ratio = cur_ratio;
                    
                    // 获取对应的假设点坐标
                    std::vector<float> h_hypo_pts(hn*2);
                    cudaMemcpy(h_hypo_pts.data(), d_hypo_pts, hn*2*sizeof(float), cudaMemcpyDeviceToHost);
                    best_point.x = h_hypo_pts[cur_max_idx*2];
                    best_point.y = h_hypo_pts[cur_max_idx*2+1];

                    // 更新最佳内点标记
                    const unsigned char* cur_inliers = &h_inliers[cur_max_idx*tn];
                    for (int i=0; i<tn; ++i) {
                        best_inlier_flags[i] = static_cast<bool>(cur_inliers[i]);
                    }
                }

                // 7. 置信度检查
                hyp_num += round_hyp_num;
                if (1 - std::pow(1 - best_ratio*best_ratio, hyp_num) > confidence) {
                    break;
                }
            }

            // ============== 新增最终投票步骤 ==============
            // 1. 准备最终假设点
            std::vector<float> h_final_hypo(1*2);
            h_final_hypo[0] = best_point.x;
            h_final_hypo[1] = best_point.y;

            // 2. 分配设备内存
            float* d_final_hypo;
            bool* d_final_inliers;
            cudaMalloc(&d_final_hypo, 1*2*sizeof(float));
            cudaMalloc(&d_final_inliers, 1*tn*sizeof(bool));
            cudaMemcpy(d_final_hypo, h_final_hypo.data(), 1*2*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_final_inliers, 0, 1*tn*sizeof(bool));

            // 3. 执行最终投票
            int* dummy_counts; // 不需要实际计数
            cudaMalloc(&dummy_counts, sizeof(int));
            launch_voting_cuda(
                d_coords, d_direct, 
                d_final_hypo, dummy_counts,
                d_final_inliers,
                h, w, tn, 1, // hn=1
                inlier_thresh
            );

            // 4. 获取最终inlier标记
            std::vector<unsigned char> final_inliers(tn);
            cudaMemcpy(final_inliers.data(), d_final_inliers, tn*sizeof(bool), cudaMemcpyDeviceToHost);
            std::vector<bool> final_inliers_bool;
            final_inliers_bool.reserve(final_inliers.size()); // 预分配空间（可选，提升效率）

            // 使用 std::transform 转换每个元素
            std::transform(
                final_inliers.begin(),
                final_inliers.end(),
                std::back_inserter(final_inliers_bool),
                [](unsigned char c) { return c != 0; } // 非零值转为 true，零为 false
            );
            // 5. 使用最终inlier进行优化
            std::vector<cv::Vec2f> directions(tn);
            for (int i=0; i<tn; ++i) {
                cv::Point2f pt = coords[i];
                directions[i] = vertex_k.at<cv::Vec2f>(pt.y, pt.x);
            }            
            output_points[k] = refinePointUsingInliersV2(coords, directions, final_inliers_bool);

            // 6. 清理资源
            cudaFree(d_final_hypo);
            cudaFree(d_final_inliers);
            cudaFree(dummy_counts);

            // ============== 释放GPU内存 ==============
            cudaFree(d_direct);
            cudaFree(d_coords);
            cudaFree(d_idxs);
            cudaFree(d_hypo_pts);
            cudaFree(d_inlier_counts);
            cudaFree(d_inliers);
        }
    }

    void ransacVotingV4(
        const cv::Mat& mask,
        const std::vector<cv::Mat>& vertex,
        std::vector<cv::Point2f>& output_points,
        int round_hyp_num,
        float inlier_thresh,
        float confidence,
        int max_iter,
        int min_num,
        int max_num 
    ) {
        std::mt19937 rng(0); // 固定随机数种子
        std::vector<cv::Point2f> coords;
        cv::findNonZero(mask, coords);

        int tn = coords.size();
        int vn = vertex.size();
        output_points.resize(vn, cv::Point2f(0, 0));

        if (tn < min_num) {
            std::cerr << "Not enough foreground points for RANSAC!" << std::endl;
            return;
        }

        // 如果前景点过多，进行下采样
        if (tn > max_num) {
            // filterForegroundPoints(coords, tn, max_num);
            tn = coords.size(); // 更新前景点数量
        }

        for (int k = 0; k < vn; ++k) {
            std::vector<cv::Point2f> hypothesis(round_hyp_num);
            std::vector<int> inlier_counts(round_hyp_num, 0);
            std::vector<std::vector<bool>> inliers(round_hyp_num, std::vector<bool>(tn, false));

            float best_ratio = 0.0f;
            cv::Point2f best_point(0, 0);
            int hyp_num = 0;
            // std::cout<< "vetex shape:"<<vertex[k].size()<<std::endl;
            for (int iter = 0; iter < max_iter; ++iter) {
                // 生成假设
                for (int r = 0; r < round_hyp_num; ++r) {
                    int idx = rng() % tn;
                    cv::Point2f pt = coords[idx];
                    cv::Vec2f offset = vertex[k].at<cv::Vec2f>(pt.y, pt.x);
                    hypothesis[r] = cv::Point2f(pt.x + offset[0], pt.y + offset[1]);
                }

                // 假设投票
                for (int i = 0; i < tn; ++i) {
                    cv::Point2f pt = coords[i];
                    cv::Vec2f offset = vertex[k].at<cv::Vec2f>(pt.y, pt.x);
                    cv::Point2f projected_pt(pt.x + offset[0], pt.y + offset[1]);

                    for (int r = 0; r < round_hyp_num; ++r) {
                        if (cv::norm(projected_pt - hypothesis[r]) < inlier_thresh) {
                            inlier_counts[r]++;
                            inliers[r][i] = true;
                        }
                    }
                }

                // 找到最佳假设
                int max_idx = std::distance(inlier_counts.begin(), std::max_element(inlier_counts.begin(), inlier_counts.end()));
                float cur_ratio = static_cast<float>(inlier_counts[max_idx]) / tn;

                if (cur_ratio > best_ratio) {
                    best_ratio = cur_ratio;
                    best_point = hypothesis[max_idx];
                }

                // 检查置信度
                hyp_num += round_hyp_num;
                if (1 - std::pow(1 - best_ratio * best_ratio, hyp_num) > confidence) {
                    break;
                }
            }

            // 使用最佳内点重新计算
            if (!inliers.empty()) {
                const auto& best_inlier_flags = inliers[std::distance(inlier_counts.begin(), std::max_element(inlier_counts.begin(), inlier_counts.end()))];
                std::vector<cv::Vec2f> directions(tn);

                for (int i = 0; i < tn; ++i) {
                    cv::Point2f pt = coords[i];
                    directions[i] = vertex[k].at<cv::Vec2f>(pt.y, pt.x);
                }

                output_points[k] = refinePointUsingInliersV2(coords, directions, best_inlier_flags);
            } else {
                output_points[k] = cv::Point2f(0, 0);
            }
        }
    }

    void ransacVotingV5(
        const cv::Mat& mask,
        const std::vector<cv::Mat>& vertex,
        std::vector<cv::Point2f>& output_points,
        int round_hyp_num,
        float inlier_thresh,
        float confidence,
        int max_iter,
        int min_num,
        int max_num,
        RansacState& state // 新增状态参数
    ) {
        std::mt19937 rng(0); // 固定随机数种子
        std::vector<cv::Point2f> coords;
        cv::findNonZero(mask, coords);

        int tn = coords.size();
        int vn = vertex.size();
        output_points.resize(vn, cv::Point2f(0, 0));

        if (tn < min_num) {
            std::cerr << "Not enough foreground points for RANSAC!" << std::endl;
            return;
        }

        // 如果前景点过多，进行下采样
        if (tn > max_num) {
            // filterForegroundPoints(coords, tn, max_num);
            tn = coords.size(); // 更新前景点数量
        }

        for (int k = 0; k < vn; ++k) {
            std::vector<cv::Point2f> hypothesis(round_hyp_num);
            std::vector<int> inlier_counts(round_hyp_num, 0);
            std::vector<std::vector<bool>> inliers(round_hyp_num, std::vector<bool>(tn, false));

            float best_ratio = 0.0f;
            cv::Point2f best_point(0, 0);
            int hyp_num = 0;

            // 如果是首次运行，初始化状态
            if (!state.is_initialized) {
                state.best_hypothesis.resize(vn, cv::Point2f(0, 0));
                state.best_inlier_counts.resize(vn, 0);
                state.best_inliers.resize(vn, std::vector<bool>(tn, false));
                state.previous_output_points.resize(vn, cv::Point2f(0, 0));
                state.is_initialized = true;
            } else {
                // 非首次运行，利用之前的状态作为初值
                hypothesis[0] = state.best_hypothesis[k];
                inlier_counts[0] = state.best_inlier_counts[k];
                inliers[0] = state.best_inliers[k];
            }

            for (int iter = 0; iter < max_iter; ++iter) {
                // 生成假设
                for (int r = (state.is_initialized ? 1 : 0); r < round_hyp_num; ++r) {
                    int idx = rng() % tn;
                    cv::Point2f pt = coords[idx];
                    cv::Vec2f offset = vertex[k].at<cv::Vec2f>(pt.y, pt.x);
                    hypothesis[r] = cv::Point2f(pt.x + offset[0], pt.y + offset[1]);
                }

                // 假设投票
                for (int i = 0; i < tn; ++i) {
                    cv::Point2f pt = coords[i];
                    cv::Vec2f offset = vertex[k].at<cv::Vec2f>(pt.y, pt.x);
                    cv::Point2f projected_pt(pt.x + offset[0], pt.y + offset[1]);

                    for (int r = 0; r < round_hyp_num; ++r) {
                        if (cv::norm(projected_pt - hypothesis[r]) < inlier_thresh) {
                            inlier_counts[r]++;
                            inliers[r][i] = true;
                        }
                    }
                }

                // 找到最佳假设
                int max_idx = std::distance(inlier_counts.begin(), std::max_element(inlier_counts.begin(), inlier_counts.end()));
                float cur_ratio = static_cast<float>(inlier_counts[max_idx]) / tn;

                if (cur_ratio > best_ratio) {
                    best_ratio = cur_ratio;
                    best_point = hypothesis[max_idx];
                    state.best_hypothesis[k] = best_point; // 更新状态
                    state.best_inlier_counts[k] = inlier_counts[max_idx];
                    state.best_inliers[k] = inliers[max_idx];
                }

                // 检查置信度
                hyp_num += round_hyp_num;
                if (1 - std::pow(1 - best_ratio * best_ratio, hyp_num) > confidence) {
                    break;
                }
            }

            // 使用最佳内点重新计算
            if (!inliers.empty()) {
                const auto& best_inlier_flags = state.best_inliers[k];
                std::vector<cv::Vec2f> directions(tn);

                for (int i = 0; i < tn; ++i) {
                    cv::Point2f pt = coords[i];
                    directions[i] = vertex[k].at<cv::Vec2f>(pt.y, pt.x);
                }

                output_points[k] = refinePointUsingInliersV2(coords, directions, best_inlier_flags);
            } else {
                output_points[k] = cv::Point2f(0, 0);
            }

            state.previous_output_points[k] = output_points[k]; // 记录输出点
        }
    }

}