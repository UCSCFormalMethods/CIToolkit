import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import json

def plot(samples):
    clean_samples = []

    for sample in samples:
        label = sample[1]
        vehs = list(set([i[0] for i in sample[0]]))

        clean_samples.append((vehs, label))

    # Load veh params
    with open("input.json") as f:
        job_json = json.load(f)

    veh_dict = job_json["vehicles"]
    print(veh_dict)

    param_samples = []

    for sample in clean_samples:
        for veh in sample[0]:
            print(veh)
            sample_params = veh_dict[veh]
            print(sample_params)
            print()

            params = (sample_params["length"], sample_params["radius"], sample_params["width"])

            param_samples.append((params, sample[1]))

    print(param_samples)

    color_map = ["red", "green", "blue"]
    marker_map = [4,5,6] # [(3, 0, 0), (3, 0, 40), (3, 0, 80)]
    # Plot

    xs = []
    ys = []
    zs = []
    cs = []
    ms = []


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for label in range(3):
        ax.set_xlabel('Length')
        ax.set_ylabel('Radius')
        ax.set_zlabel('Width')

        ax.set_xlim(xmin=100,xmax=300)
        ax.set_ylim(ymin=50,ymax=140)
        ax.set_zlim(zmin=65,zmax=215)

        print(xs)

        xs = [sample[0][0] for sample in param_samples if sample[1] == label]
        ys = [sample[0][1] for sample in param_samples if sample[1] == label]
        zs = [sample[0][2] for sample in param_samples if sample[1] == label]
        cs = [color_map[sample[1]] for sample in param_samples if sample[1] == label]

        ax.scatter(xs, ys, zs, c=cs, marker=marker_map[label], alpha=1, s=40)

    print(xs)
    print(cs)

    label = "all"

    def rotate(angle):
        ax.view_init(azim=angle)

    plt.savefig('plan_3d_visuals/Label_all.png', dpi=750, bbox_inches='tight')

    # rot_animation = matplotlib.animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 0.5), interval=200)

    # rot_animation.save('plan_3d_visuals/Label_' + str(label) + '.gif', writer='pillow', fps=30)

    # plt.close("all")


if __name__ == '__main__':
    SAMPLES =   [
                ([['28', '4'], ['48', '0'], ['48', '2'], ['28', '3'], ['48', '1']], 0),
                ([['40', '3'], ['48', '0'], ['48', '2'], ['23', '4'], ['8', '1']], 1),
                ([['28', '4'], ['44', '1'], ['48', '0'], ['41', '3'], ['43', '2']], 2)
                ]

    plot(SAMPLES)