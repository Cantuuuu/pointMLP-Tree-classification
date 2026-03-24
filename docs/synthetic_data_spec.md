# Especificaciones para Datos Sinteticos LiDAR

## Comparacion justa: modelo entrenado con datos reales (NEON/IDTReeS) vs sinteticos

---

## 1. Clases y mapeo

Usamos **3 clases**. El mapeo es directo:

| Clase sintetica | Clase del modelo | Especies reales agrupadas                          |
|-----------------|------------------|-----------------------------------------------------|
| **Pine**        | PINE             | *Pinus palustris*                                   |
| **Oak**         | OAK              | *Quercus alba, Q. coccinea, Q. laevis, Q. rubra*   |
| **Deciduous**   | OTHER            | *Acer rubrum, Amelanchier laevis, Nyssa sylvatica*  |

---

## 2. Cantidad de arboles necesaria

El dataset real tiene **767 arboles** con esta distribucion:

| Clase   | Train | Val | Test | Total   |
|---------|-------|-----|------|---------|
| PINE    | 152   | 19  | 19   | **190** |
| OAK     | 283   | 35  | 35   | **353** |
| OTHER   | 178   | 23  | 23   | **224** |
| **Total** | **613** | **77** | **77** | **767** |

Se necesitan **minimo 767 arboles sinteticos** con distribucion similar:

- **~190 Pine**
- **~353 Oak**
- **~224 Deciduous**

Con la generacion actual de 250 arboles balanceados, se necesitan **4 generaciones** (~1,000 arboles).
Nosotros recortamos para igualar la distribucion exacta.

**Importante:** Variar el `seed_base` entre generaciones para tener diversidad real. Que no sean copias de los mismos arboles.

---

## 3. Formato de entrega

El formato actual funciona perfecto. Por cada generacion necesitamos:

- **`points.laz`** con:
  - `point_source_id` = tree_id (0=suelo, 1-N=arboles)
  - `user_data` = species_id (1=Deciduous, 2=Oak, 3=Pine)
- **`metadata.json`** con la lista de arboles y su especie

---

## 4. Densidad de puntos por arbol

Este es el parametro mas importante para la comparacion justa.

### Datos reales (NEON/IDTReeS)

Los archivos LAS reales **no contienen metadata de altitud de vuelo**.
Lo que si sabemos directamente de los datos:

| Metrica                   | Valor                          |
|---------------------------|--------------------------------|
| Puntos por arbol          | **5,500 - 15,800** (media ~9,500) |
| Altura de arboles         | 2.4 - 39.3 m                  |
| Sitios                    | MLBS (Virginia), OSBS (Florida) |
| Software de procesamiento | rlas (R package)               |
| Formato                   | LAS 1.3, Point Format 1       |

### Datos sinteticos (config actual)

| Metrica              | Valor                              |
|----------------------|------------------------------------|
| Puntos por arbol     | **1,531 - 66,445** (media ~13,700) |
| Altitud simulada     | 20m                                |
| PRF                  | 240,000                            |
| FOV                  | 70 grados                          |
| Max returns          | 5                                  |
| Range noise sigma    | 0.005                              |

### Comparacion

La densidad sintetica actual (media ~13,700 pts/arbol) esta en el mismo orden
de magnitud que la real (~9,500 pts/arbol). **La config actual es aceptable.**

Nosotros submuestreamos todos los arboles a **1,024 puntos** antes de entrenar,
asi que ambos datasets pasan por el mismo cuello de botella. La diferencia de
densidad original se mitiga con el submuestreo.

Si quieres ajustar para mayor precision, apunta a una densidad promedio de
**~8,000-12,000 puntos por arbol** para estar mas cerca del dato real.

---

## 5. Solo coordenadas XYZ

Aunque el LAZ tenga mas campos (intensity, return_number, classification),
**solo usaremos coordenadas X, Y, Z** para igualar con los datos reales.
No hay que ajustar nada en la generacion, nosotros filtramos al preprocesar.

---

## 6. Resumen rapido

> Necesito:
> 1. **~1,000 arboles totales**: ~190 Pine, ~353 Oak, ~224 Deciduous (o mas y nosotros recortamos)
> 2. **Mismo formato** que ya tienes (LAZ con point_source_id + metadata.json)
> 3. **Seeds diferentes** entre generaciones para variabilidad
> 4. **Densidad similar a los datos reales**: ~8,000-12,000 puntos por arbol promedio (tu config actual da ~13,700, esta cerca y es aceptable)
> 5. Lo demas esta perfecto como esta
