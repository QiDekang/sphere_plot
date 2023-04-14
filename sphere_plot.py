import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 转换为球面坐标的函数
def get_sphere_data(r, u, v):

    ## θ为有向线段OP与z轴正向的夹角。θ∈[0, π]。
    ## v, z轴的变化
    ## φ为从正z轴来看自x轴按逆时针方向转到OM所转过的角，这里M为点P在xOy面上的投影。φ∈[0,2π] 
    ## u，xy平面的变化


    ## x=rsinθcosφ.
    x = r * np.outer(np.cos(u), np.sin(v))
    ## y=rsinθsinφ.
    y = r * np.outer(np.sin(u), np.sin(v))
    ## z=rcosθ.
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z



def get_horizontal_angle(business_district_num, day_num, gap_size_ratio):

    ## 总柱子数量
    Bar_num = business_district_num * day_num
    ## 总间隔数量
    gap_num = Bar_num

    ## 计算每一个间隔对应的角度大小
    gap_angle = 2 * np.pi / ((1 + (1 / gap_size_ratio)) * gap_num)
    bar_angle = gap_angle * (1 / gap_size_ratio)
    #print('gap_angle', gap_angle)
    #print('bar_angle', bar_angle)

    return bar_angle, gap_angle


def get_vertical_angle(min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical):

    ## 上半球
    ##  上半球基础作图区域：0 ~ 0.5 * np.pi --> (1 - max_latitude_ratio) * np.pi ~ 0.5 * np.pi
    ## 上半球作图区域为 (1 - max_latitude_ratio) * np.pi ~ (0.5 - min_latitude_ratio) * np.pi
    ## 作图区域
    area_lower_boundary = (0.5 - min_latitude_ratio) * np.pi
    area_upper_boundary = (1 - max_latitude_ratio) * np.pi
    area_total_len = area_lower_boundary - area_upper_boundary

    ## 作图区域内，每一个图块和间隔的角度大小
    block_gap_num = block_num - 1
    block_gap_angle = area_total_len / (block_gap_num + block_num * (1 / gap_size_ratio_vertical))
    block_angle = block_gap_angle * (1 / gap_size_ratio_vertical)

    return block_angle, block_gap_angle, area_lower_boundary, area_upper_boundary, area_total_len
    

def get_vertical_angle_upper_different(min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical):

    ## 上半球
    ##  上半球基础作图区域：0 ~ 0.5 * np.pi --> (1 - max_latitude_ratio) * np.pi ~ 0.5 * np.pi
    ## 上半球作图区域为 (1 - max_latitude_ratio) * np.pi ~ (0.5 - min_latitude_ratio) * np.pi
    ## 作图区域
    area_lower_boundary = (0.5 - min_latitude_ratio) * np.pi
    area_upper_boundary = (1 - max_latitude_ratio) * np.pi
    area_total_len = area_lower_boundary - area_upper_boundary

    ## 作图区域内，每一个图块和间隔的角度大小
    ### 用最下面一个柱子代表80，上面一个柱子代表5。总共5个柱子
    block_num_new = 5
    block_gap_num = block_num_new - 1
    block_gap_angle = area_total_len / (block_gap_num + block_num_new * (1 / gap_size_ratio_vertical))
    block_angle = block_gap_angle * (1 / gap_size_ratio_vertical)

    return block_angle, block_gap_angle, area_lower_boundary, area_upper_boundary, area_total_len
    

def get_vertical_angle_lower_hemispheres(min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical):

    ## 上半球
    ##  上半球基础作图区域：0 ~ 0.5 * np.pi --> (1 - max_latitude_ratio) * np.pi ~ 0.5 * np.pi
    ## 上半球作图区域为 (1 - max_latitude_ratio) * np.pi ~ (0.5 - min_latitude_ratio) * np.pi
    ## 作图区域
    area_lower_boundary = (0.5 + min_latitude_ratio) * np.pi
    area_upper_boundary = max_latitude_ratio * np.pi
    area_total_len = area_upper_boundary - area_lower_boundary

    ## 作图区域内，每一个图块和间隔的角度大小
    block_gap_num = block_num - 1
    block_gap_angle = area_total_len / (block_gap_num + block_num * (1 / gap_size_ratio_vertical))
    block_angle = block_gap_angle * (1 / gap_size_ratio_vertical)

    return block_angle, block_gap_angle, area_lower_boundary, area_total_len

def get_end_len(is_half, business_district_num):
    if is_half == True:
        end_len = int(business_district_num / 2)
    else:
        end_len = business_district_num
    
    return end_len

def get_part_data(is_half, start_id, end_len, data):
    if is_half == True:
        end_id = start_id + end_len
        data = data.iloc[start_id:end_id, :]
        data = data.reset_index(drop=True)  # 重设索引
    else:
        data = data
    
    return data


def plot_upper_hemispheres(ax, upper_data, business_district_num, day_num, r, gap_size_ratio, min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical, colour_upper, is_half, start_id):

    bar_angle, gap_angle = get_horizontal_angle(business_district_num, day_num, gap_size_ratio)
    block_angle, block_gap_angle, area_lower_boundary, area_upper_boundary, area_total_len = get_vertical_angle(min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical)

    end_len = get_end_len(is_half, business_district_num)
    upper_data = get_part_data(is_half, start_id, end_len, upper_data)

    ## 画上半球
    for i in range(0, end_len):
        for j in range(0, day_num):
            bar_id = i * day_num + j
            #print('bar_id', bar_id)


            ## 平面φ角
            u = [bar_id * (bar_angle + gap_angle), bar_id * (bar_angle + gap_angle) + bar_angle]
            #print('u', u)
                
            ## 柱状图长度
            bar_len = (upper_data.iloc[i, j] / 100) * area_total_len

            ## 仰角分块展示，每块代表大小为10
            ### 向下取整
            bar_len_num = int(upper_data.iloc[i, j] / 10)
            
            
            ### 先画10的整数倍的数据
            for k in range(0, bar_len_num):
                v = [area_lower_boundary - (k * (block_angle + block_gap_angle) + block_angle), area_lower_boundary - k * (block_angle + block_gap_angle)]
                plot_one_block(r, u, v, colour_upper)
                

            ### 再画不满10的数据
            v = [area_lower_boundary - bar_len, area_lower_boundary - bar_len_num * (block_angle + block_gap_angle)]
            plot_one_block(r, u, v, colour_upper)
            
            


def plot_upper_hemispheres_different(ax, upper_data, business_district_num, day_num, r, gap_size_ratio, min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical, colour_upper, is_half, start_id, colour_upper_dark):

    bar_angle, gap_angle = get_horizontal_angle(business_district_num, day_num, gap_size_ratio)
    block_angle, block_gap_angle, area_lower_boundary, area_upper_boundary, area_total_len = get_vertical_angle_upper_different(min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical)

    end_len = get_end_len(is_half, business_district_num)
    upper_data = get_part_data(is_half, start_id, end_len, upper_data)

    ## 画上半球
    for i in range(0, end_len):
        for j in range(0, day_num):
            bar_id = i * day_num + j
            #print('bar_id', bar_id)


            ## 平面φ角
            u = [bar_id * (bar_angle + gap_angle), bar_id * (bar_angle + gap_angle) + bar_angle]
            #print('u', u)

            ### 先画第一个块，代表80
            if upper_data.iloc[i, j] < 80:
                first_len = (upper_data.iloc[i, j] / 80) * block_angle
                v = [area_lower_boundary - first_len, area_lower_boundary]
                plot_one_block(r, u, v, colour_upper_dark)
            else:
                v = [area_lower_boundary - block_angle, area_lower_boundary]
                plot_one_block(r, u, v, colour_upper_dark)

                ### 再画其他块
                ## 柱状图长度
                #bar_len = ((upper_data.iloc[i, j] - 80) / 80) * (area_total_len - block_angle - block_gap_angle)

                area_lower_boundary_new = area_lower_boundary - block_angle - block_gap_angle
                ### 向下取整
                #### 每个柱子代表5，总共4个柱子，共20。
                bar_len_num = int((upper_data.iloc[i, j] - 80) / 5)
                
                ### 先画10的整数倍的数据
                for k in range(0, bar_len_num):
                    v = [area_lower_boundary_new - (k * (block_angle + block_gap_angle) + block_angle), area_lower_boundary_new - k * (block_angle + block_gap_angle)]
                    plot_one_block(r, u, v, colour_upper)
                    

                ### 再画不满10的数据
                remaining_len = ((upper_data.iloc[i, j] - 80) - int((upper_data.iloc[i, j] - 80) / 5) * 5) / 5
                remaining_bar = remaining_len * block_angle
                v = [area_lower_boundary_new - bar_len_num * (block_angle + block_gap_angle) - remaining_bar, area_lower_boundary_new - bar_len_num * (block_angle + block_gap_angle)]
                plot_one_block(r, u, v, colour_upper)
            

def plot_equator(ax, business_district_num, day_num, gap_size_ratio, min_latitude_ratio, colour_list, is_half, start_id):

    bar_angle, gap_angle = get_horizontal_angle(business_district_num, day_num, gap_size_ratio)

    end_len = get_end_len(is_half, business_district_num)

    ## 赤道图例
    for i in range(0, end_len):

        colour_id = i % 7
        colour = colour_list[colour_id]

        for j in range(0, day_num):
            bar_id = i * day_num + j
            if j == 0:
                u = [bar_id * (bar_angle + gap_angle), bar_id * (bar_angle + gap_angle) + bar_angle]
            else:
                u = [bar_id * (bar_angle + gap_angle) - gap_angle - 0.3 * gap_angle, bar_id * (bar_angle + gap_angle) + bar_angle]
            #赤道作图区域
            ## (0.5-min_latitude_ratio) * np.pi ~ (0.5+min_latitude_ratio) * np.pi
            v = [(0.5 - min_latitude_ratio * gap_size_ratio) * np.pi, (0.5 + min_latitude_ratio * gap_size_ratio) * np.pi]
            plot_one_block(r, u, v, colour)


def get_type_data(type_id, business_district_num, time_num):

    lower_data = pd.DataFrame(np.random.randint(0,100,size=(business_district_num, time_num)))
    for i in range(0, len(lower_data)):
        lower_data.iloc[i, :] = 100 * lower_data.iloc[i, :] / sum(lower_data.iloc[i, :])
    lower_data['type'] = type_id

    return lower_data


def plot_lower_hemispheres(ax, lower_data, business_district_num, type_num, time_num, r, gap_size_ratio, min_latitude_ratio, max_latitude_ratio, gap_size_ratio_vertical, colour_lower_list, is_half, start_id, is_lower_first_difference_colour):

    bar_angle, gap_angle = get_horizontal_angle(business_district_num, type_num, gap_size_ratio)
    block_angle, block_gap_angle, area_lower_boundary, area_total_len = get_vertical_angle_lower_hemispheres(min_latitude_ratio, max_latitude_ratio, time_num, gap_size_ratio_vertical)
    
    end_len = get_end_len(is_half, business_district_num)

    ## 画下半球
    for i in range(0, end_len):
        for j in range(0, type_num):
            ## 同一商圈、同一月，不同时间段的平面角相同。
            current_lower_data = lower_data[lower_data['type'] == j]
            #print('current_lower_data', current_lower_data)
            ## 取从start_id开始的部分数据
            current_lower_data = get_part_data(is_half, start_id, end_len, current_lower_data)
            bar_id = i * type_num + j
            ## 平面φ角
            u = [bar_id * (bar_angle + gap_angle), bar_id * (bar_angle + gap_angle) + bar_angle]

            ## 暂存的当前方块的下界
            current_block_lower_boundary = area_lower_boundary
            ## 画不同时间段
            for k in range(0, time_num):
                colour_lower = colour_lower_list[k]
                ### 当前方块角度大小是等分方块角度大小的倍数
                current_bar_len_ratio = current_lower_data.iloc[i, k] / (100 / time_num)
                current_bar_len = current_bar_len_ratio * block_angle

                #v = [current_block_lower_boundary, current_block_lower_boundary + current_bar_len]
                # 弧线
                ## 因数据有效位数问题，弧线间会有缝隙
                v = np.linspace(current_block_lower_boundary, current_block_lower_boundary + current_bar_len, 5)
                current_block_lower_boundary = current_block_lower_boundary + current_bar_len + block_gap_angle

                ## plot
                #plot_one_block(r, u, v, colour_lower)
                if is_lower_first_difference_colour == False and j == 0:
                    #print('is_lower_first_difference_colour')
                    plot_one_block(r, u, v, '#F43F5E')
                else:
                   plot_one_block(r, u, v, colour_lower) 



def plot_background(ax, r):

    # 背景球面的半径略小
    background_r = r * 0.75

    # Make data
    ## θ为有向线段OP与z轴正向的夹角。θ∈[0, π]。
    v = np.linspace(0.2 * np.pi, 0.8 * np.pi, 10)
    ## φ为从正z轴来看自x轴按逆时针方向转到OM所转过的角，这里M为点P在xOy面上的投影。φ∈[0,2π] 
    u = np.linspace(0, 2 * np.pi, 10)

    x, y, z = get_sphere_data(background_r, u, v)

    # Plot the surface
    ax.plot_surface(x, y, z, color = 'white', shade=False)
    ax.grid(False)


def plot_one_block(r, u, v, colour):

    x, y, z = get_sphere_data(r, u, v)
    ax.plot_surface(x, y, z, color = colour, shade=False)
    ax.grid(False)
    #ax.set_box_aspect([1,1,1])
    

####################
if __name__ == '__main__':

    # 输入参数
    ## 半径
    r = 30
    ## 商圈数量
    business_district_num = 32
    ## 数据天数
    day_num = 7
    ## 下半球柱状图数量（月数）
    type_num = 2
    ## 下半球堆积柱状图分割数量（一天分几个时间段）
    time_num = 6
    ## 间隔大小与柱状图大小的比例
    gap_size_ratio = 0.8

    ## 纵向方块数量
    block_num = 10
    ## 纵向间隔大小与纵向每个块大小的比例
    gap_size_ratio_vertical = 0.2

    ## 纬度最高限制
    max_latitude_ratio = 0.8
    ## 纬度最低限制
    ### 赤道作为图例
    min_latitude_ratio = 0.015
    ## 调节赤道图例与上下半球的间隔
    gap_size_ratio_equator = 0.6

    ## 3D图视角
    #ax.elev = 0
    #ax.azim = 270  # xz view
    #ax.elev = 0
    #ax.azim = 0    # yz view
    #ax.elev = 0
    #ax.azim = -90  # xy view
    elev = 0
    azim = 90


    ## 颜色
    colour_list = ['#DC2626', '#EA580C', '#EAB308', '#16A34A', '#0891B2', '#2563EB', '#9333EA']
    colour_upper = '#3B82F6'
    colour_upper_dark = '#1E40AF'
    colour_lower = '#F43F5E'
    colour_lower_list = ['#9F1239', '#BE123C', '#E11D48', '#F43F5E', '#FB7185', '#FDA4AF']
    colour_background = 'white'


    # 仅展示当前半球数据，避免被其他数据干扰
    is_half = True
    ## 半球数据开始节点
    start_id = 0


    # 下层第一列柱子是否使用不同的颜色
    is_lower_first_difference_colour = False
    # 上半球，第一个柱子代表多少
    #first_represented_num = 80

    # random data
    ## data，柱状图对应的值，二维数组, business_district_num * day_num
    upper_data = pd.DataFrame(np.random.randint(0,100,size=(business_district_num, day_num)))
    lower_data_1 = get_type_data(0, business_district_num, time_num)
    lower_data_2 = get_type_data(1, business_district_num, time_num)
    lower_data_3 = get_type_data(2, business_district_num, time_num)
    lower_data = pd.concat([lower_data_1, lower_data_2, lower_data_3])
    '''
    # real data
    save_floder = 'D:/数字生活/算法/商圈洞察场景/客群洞察/data/sphere_data/'
    upper_data = pd.read_csv(save_floder + 'customer_flow_wangfujing.csv', index_col=False)
    lower_data = pd.read_csv(save_floder + 'lower_data.csv', index_col=False)
    '''
    print(upper_data)
    print(lower_data)

    
    # 画图
    # 画图
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.elev = elev
    ax.azim = azim
    ## 画背景
    #plot_background(ax, r)
    ## 画上半球
    #plot_upper_hemispheres(ax, upper_data, business_district_num, day_num, r, gap_size_ratio, min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical, colour_upper, is_half, start_id)
    # 上层第一个柱子是否使用不同的颜色，并用一个柱子代表80
    plot_upper_hemispheres_different(ax, upper_data, business_district_num, day_num, r, gap_size_ratio, min_latitude_ratio, max_latitude_ratio, block_num, gap_size_ratio_vertical, colour_upper, is_half, start_id, colour_upper_dark)
    ## 画赤道
    plot_equator(ax, business_district_num, day_num, gap_size_ratio_equator, min_latitude_ratio, colour_list, is_half, start_id)
    ## 画下半球
    plot_lower_hemispheres(ax, lower_data, business_district_num, type_num, time_num, r, gap_size_ratio, min_latitude_ratio, max_latitude_ratio, gap_size_ratio_vertical, colour_lower_list, is_half, start_id, is_lower_first_difference_colour)
    plt.axis('off')  #  关闭所有坐标轴
    plt.tight_layout()
    plt.show()
    
