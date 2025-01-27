from matplotlib import pyplot as plt


class Visualizer:

    def __init__(self) -> None:
        pass

    def plot_point_cloud(x, y, z, color='SteelBlue', marker_size=20, marker_shape=10) -> None:
        """
        Plots a 3D scatter plot with customizable marker properties.

        Args:
            x (list or array): The X-coordinates of the points.
            y (list or array): The Y-coordinates of the points.
            z (list or array): The Z-coordinates of the points.
            marker_color (str or list): The color of the markers. Can be a single color (e.g., 'red') or a list of colors for each point.
            marker_size (int or list): The size of the markers. Can be a single size or a list of sizes for individual markers.
            marker_shape (str): The shape of the markers (e.g., 'o' for circle, '^' for triangle, 's' for square, etc.).
        Returns:
            None: Displays the 3D scatter plot.

        Raises:
            ValueError: If the lengths of x, y, and z do not match.
        """
        if len(x) != len(y) or len(y) != len(z):
            raise ValueError("The lengths of x, y, and z must match.")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        ax.scatter(x, y, z, c=color, s=marker_size, marker=marker_shape)

        # Display the plot
        plt.show()
