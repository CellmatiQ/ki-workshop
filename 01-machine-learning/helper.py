import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_decision_boundary(X, Y, knn):   
    # Größe der Abbildung festlegen
    plt.figure(figsize=(10,10))

    # Abstand der zu prüfenden Punkte festlegen
    h = .02

    # Die Entscheidungsgrenze einzeichnen. Jeder überprüfte Punkt bekommt eine Farbe.
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Farben für die Darstellung festlegen
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Ergebnis in einen matplotlib-Plot verwandeln
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.set_cmap(plt.cm.Paired)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Punkte des Trainingsdatensatzes einzeichnen
    plt.scatter(X[:,0], X[:,1],c=Y, cmap=cmap_bold, edgecolor='gray')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()