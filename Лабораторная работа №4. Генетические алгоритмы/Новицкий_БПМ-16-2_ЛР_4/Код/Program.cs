using System;
using System.Collections.Generic;
using System.Linq;

namespace Genetic_algoritm
{
    class Genetic_algorithm
    {
        // Инициализируем переменные
        // Рассмотрим функцию Экли
        int count_of_individuals = 1000; // Количество особей в популяции
        double min_x = -5; // Минимальное значение х
        double max_x = 1000; // Максимальное значение x
        double min_y = -1000; // Минимальное значение y
        double max_y = 5; // Максимальное значение y
        double mutation_rate = 0.05; // Вероятность мутации для каждой особи
        double mutation_x = 0; // Начальное значение мутации по x
        double mutation_y = 0; // Начальное значение мутации по y
        int period_change_mutation_values = 10; // Период для изменения (уменьшения) пределов мутации
        double mean_change_mutation_values = 0.5; // Параметр изменения пределов мутации
        double death_rate = 0.05; // Вероятность смертности (умирают только худшие особи)
        int count_of_eras = 1000; // Количество эпох работы генетического алгоритма

        List<double> x_population = new List<double>(); // Значения параметров х
        List<double> y_population = new List<double>(); // Значения параметров у
        List<double> f = new List<double>(); // Значения функции приспособленности
        List<double> p = new List<double>(); // Значения вероятностей выбора i значения для дальнейшего отбора (селекции) (плотность распределения)
        List<double> P = new List<double>(); // Значения первообразных от вероятностей (значения функции распределения)

        double min_value = double.MaxValue; // Лучшее (минимальное значение из всех эпох обучения)
        int era_min_value = -1; // Эпоха, в которое было достигнуто минимальное значение
        double min_value_x = 0; // Значение переменной x для наименьшего значения функции
        double min_value_y = 0; // Значение переменной y для наименьшего значения функции
        List<double> mas_min_value = new List<double>(); // Список минимальных значений на каждой эпохе обучения

        // Стандартный конструктор
        public Genetic_algorithm(int count_of_individuals_, int count_of_eras_, double mutation_rate_, double death_rate_, int period_change_mutation_values_, double mean_change_mutation_values_)
        {
            count_of_individuals = count_of_individuals_;
            count_of_eras = count_of_eras_;
            mutation_rate = mutation_rate_;
            death_rate = death_rate_;
            period_change_mutation_values = period_change_mutation_values_;
            mean_change_mutation_values = mean_change_mutation_values_;
        }
        // Установка граничных значений
        public void Set_boundary_values(double min_x_, double max_x_, double min_y_, double max_y_)
        {
            min_x = min_x_;
            max_x = max_x_;
            min_y = min_y_;
            max_y = max_y_;

            mutation_x = (max_x - min_x) / count_of_individuals;
            mutation_y = (max_y - min_y) / count_of_individuals;

            Get_random_start_population();
            //Get_periodic_start_population(); // Получаем начальную популяцию
        }
        // Получение начальной популяции случайным образом в заданных пределах
        public void Get_random_start_population()
        {
            var rnd = new Random();
            for (int i = 0; i < count_of_individuals; i++)
            {
                x_population.Add(rnd.NextDouble() * (max_x - min_x) + min_x);
                y_population.Add(rnd.NextDouble() * (max_y - min_y) + min_y);
            }
        }
        // Получение начальной популяции равномерно с заданным шагом
        public void Get_periodic_start_population()
        {
            double step_x = (max_x - min_x) / count_of_individuals;
            double step_y = (max_y - min_y) / count_of_individuals;

            int counter = 0;
            for (double x = min_x; x <= max_x; x = x + step_x)
            {
                x_population.Add(x);
                counter++;
            }
            counter = 0;
            for (double y = min_y; y <= max_y; y = y + step_y)
            {
                y_population.Add(y);
                counter++;
            }
        }
        // Вывод популяции на экран
        public void Print_value_population()
        {
            for (int i = 0; i < f.Count; i++)
            {
                Console.WriteLine("{0}, {1}, {2}", x_population[i], y_population[i], f[i]);
            }
        }
        // Вывод отсортированной популяции на экран
        public void Print_sort_value_population()
        {
            List<double> f_copy = new List<double>(f);
            f_copy.Sort();
            for (int i = 0; i < f.Count; i++)
            {
                Console.WriteLine("{0}", f[i]);
            }
            Console.WriteLine();
        }
        // Вывод лучших значений особей по каждой эпохе
        public void Print_min_values()
        {
            Console.WriteLine("mas_min_values = ");
            for(int i=0; i<mas_min_value.Count; i++)
            {
                Console.WriteLine("Era = {0}. Value = {1}", i + 1, mas_min_value[i]);
            }
            Console.WriteLine();
            Console.WriteLine("Min value from all eras = {0}", min_value);
            Console.WriteLine("Min x = {0}", min_value_x);
            Console.WriteLine("Min y = {0}", min_value_y);
            Console.WriteLine("Era min value = {0}", era_min_value);
        }
        // Вычисление значений популяции
        public void Calculation_population_values()
        {
            f.Clear();
            double[] helper_mas = new double[2];
            for (int i = 0; i < count_of_individuals; i++)
            {
                helper_mas[0] = x_population[i];
                helper_mas[1] = y_population[i];

                //f.Add(Rastrigin_function(helper_mas)); // Good work
                //f.Add(Ekli_function(x_population[i], y_population[i])); // Good work
                //f.Add(Sphere_function(helper_mas));
                //f.Add(Rozenbroke_function(helper_mas));
                //f.Add(Bill_function(x_population[i], y_population[i])); // Good work
                //f.Add(Goldman_Price_function(x_population[i], y_population[i]));
                //f.Add(Byte_function(x_population[i], y_population[i])); // Good work
                //f.Add(Bukin_function(x_population[i], y_population[i]));
                f.Add(Levi_function(x_population[i], y_population[i]));
            }

            //for (int i = 0; i < count_of_individuals; i++)
            //{
            //    //f.Add(Ekli_function(x_population[i], y_population[i]));
            //    f.Add(Bill_function(helper_mas[0], helper_mas[1]));
            //}

            //Console.WriteLine("len(f) = {0}", f.Count);
        }
        // Нахождение минимального значения из всех особей
        public void Get_population_min_value(int value_era)
        {
            double min_value_in_era = double.MaxValue;
            double min_value_x_in_era = 0;
            double min_value_y_in_era = 0;


            for (int i=0; i<f.Count; i++)
            {
                if(f[i]<min_value_in_era)
                {
                    min_value_in_era = f[i];
                    min_value_x_in_era = x_population[i];
                    min_value_y_in_era = y_population[i];
                }
            }
            mas_min_value.Add(min_value_in_era);
            if(min_value_in_era<min_value)
            {
                min_value = min_value_in_era;
                min_value_x = min_value_x_in_era;
                min_value_y = min_value_y_in_era;
                era_min_value = value_era;
            }
        }
        // Смерть особей популяции
        public void Death()
        {
            int i_max = -1; // Индекс с максимальным значением элемента
            double max = double.MinValue;
            for(int i=0; i<(int)count_of_individuals*death_rate; i++)
            {
                i_max = -1;
                max = double.MinValue;
                for (int j=0; j<f.Count; j++)
                {
                    if(f[j] > max)
                    {
                        max = f[j];
                        i_max = j;
                    }
                }
                //Console.WriteLine("i_max = {0}", i_max);
                //Console.WriteLine("f.Count = {0}", f.Count);
                //Console.WriteLine("x_population.Count = {0}", x_population.Count);
                //Console.WriteLine("y_population.Count = {0}", y_population.Count);

                x_population.RemoveAt(i_max);
                y_population.RemoveAt(i_max);

                f.RemoveAt(i_max);
            }
        }
        // Отбор (селекция) с помощью метода сигмы-отсечения
        public void Selection_sigma_clipping()
        {
            double sigma = 0; // Среднеквадратичное отклонение значения целевой функции
            for (int i = 0; i < f.Count; i++)
            {
                sigma = sigma + Math.Pow(f[i] - f.Average(), 2);
            }
            sigma = sigma / f.Count;
            sigma = Math.Sqrt(sigma);

            List<double> F_x = new List<double>();
            //Console.WriteLine("F(x) =");
            for(int i=0; i<f.Count; i++)
            {
                F_x.Add(1 + (f[i] - f.Average()) / (2 * sigma));
                //Console.WriteLine("{0}", F_x[i]);
            }

            p.Clear();
            for(int i=0; i<f.Count; i++)
            {
                p.Add(F_x[i] / F_x.Sum());
            }

            //Console.WriteLine("P =");
            P.Clear();
            double sum = 0;
            for(int i = 0; i < f.Count; i++)
            {
                sum = sum + p[i];
                P.Add(sum);
                //Console.WriteLine(P[i]);
            }

            //Console.WriteLine("P.Count = {0}", P.Count);
            //Console.WriteLine("x_population.Count = {0}", x_population.Count);
            //Console.WriteLine("y_population.Count = {0}", y_population.Count);

            var r = new Random();
            while (x_population.Count < count_of_individuals)
            {
                // Выбор родителя №1
                int number_parent_1 = 0;
                int number_parent_2 = 0;
                while ((P[number_parent_1] < r.NextDouble()) && (number_parent_1 < P.Count))
                {
                    number_parent_1++;
                }

                // Выбор родителя №2
                while ((P[number_parent_2] < r.NextDouble()) && (number_parent_2 < P.Count))
                {
                    number_parent_2++;
                }

                // Создание новой особи
                double x_gen_percentage = r.NextDouble();
                double y_gen_percentage = r.NextDouble();
                x_population.Add(x_gen_percentage * x_population[number_parent_1] + (1 - x_gen_percentage) * x_population[number_parent_2]);
                y_population.Add(y_gen_percentage * y_population[number_parent_1] + (1 - y_gen_percentage) * y_population[number_parent_2]);
            }

            //Console.WriteLine("x_population.Cout = {0}", x_population.Count);
            //Console.WriteLine("y_population.Cout = {0}", y_population.Count);
        }
        // Отбор (селекция) случайным образом
        public void Random_selection()
        {
            var r = new Random();

            while(x_population.Count < count_of_individuals)
            {
                int parent_1 = (int)(r.NextDouble() * count_of_individuals * (1 - death_rate) - 1);
                int parent_2 = (int)(r.NextDouble() * count_of_individuals * (1 - death_rate) - 1);
                //Console.WriteLine("parent_1 = {0}", parent_1);
                //Console.WriteLine("parent_2 = {0}", parent_2);
                //Console.WriteLine("x_population.Count = {0}", x_population.Count);
                //Console.WriteLine("y_population.Count = {0}", y_population.Count);

                double value_rand = r.NextDouble();
                x_population.Add(value_rand * x_population[parent_1] + (1 - value_rand) * x_population[parent_2]);
                value_rand = r.NextDouble();
                y_population.Add(value_rand * y_population[parent_1] + (1 - value_rand) * y_population[parent_2]);
            }
        }
        // Метод, организующий мутацию
        public void Mutation()
        {
            List<int> index = new List<int>(); // Значения индексов
            List<int> mutation_index_x = new List<int>(); // Значения индексов x для мутации
            List<int> mutation_index_y = new List<int>(); // Значения индексов y для мутации


            for (int i = 0; i < x_population.Count; i++)
            {
                index.Add(i);
            }

            // Выбор значений x из популяции для дальнейшей мутации
            var r = new Random();
            int value_index = 0;
            for (int i = 0; i < x_population.Count * mutation_rate; i++)
            {
                value_index = (int)(r.NextDouble() * index.Count);
                mutation_index_x.Add(value_index);
                index.RemoveAt(value_index);
            }

            index.Clear();
            for (int i = 0; i < x_population.Count; i++)
            {
                index.Add(i);
            }

            for (int i = 0; i < y_population.Count * mutation_rate; i++)
            {
                value_index = (int)(r.NextDouble() * index.Count);
                mutation_index_y.Add(value_index);
                index.RemoveAt(value_index);
            }

            // Процесс мутации
            double mutation_value_x = 0; // Значение, на которое происходит мутация особи по x
            double mutation_value_y = 0; // Значение, на которое происходит мутация особи по y
            for (int i = 0; i < mutation_index_x.Count; i++)
            {
                mutation_value_x = r.NextDouble() * mutation_x - mutation_x / 2;
                mutation_value_y = r.NextDouble() * mutation_y - mutation_y / 2;
                x_population[mutation_index_x[i]] = x_population[mutation_index_x[i]] + mutation_value_x;
                y_population[mutation_index_y[i]] = y_population[mutation_index_y[i]] + mutation_value_y;
            }
        }
        // Обучающая функция
        public void Learning_function()
        {
            for(int value_era = 0; value_era<count_of_eras; value_era++)
            {
                Console.WriteLine("Era of learning №{0}", value_era + 1);

                Calculation_population_values(); // Вычисляем значения начальной популяции
                //Print_value_population();
                Get_population_min_value(value_era);

                //Print_sort_value_population();
                Death(); // Убиваем неприспособившиеся особи
                //Print_sort_value_population();
                //Selection_sigma_clipping(); // Рождение новых особей с помощью метода сигмы-отсечения
                Random_selection();
                Mutation(); // Мучация части популяции

                if ((value_era + 1) % period_change_mutation_values == 0)
                {
                    mutation_x = mutation_x * mean_change_mutation_values;
                    mutation_y = mutation_y * mean_change_mutation_values;
                }
            }
            Print_min_values();
        }
        // Получить x лучших значений популяции
        public void Set_the_best_values(double[,] values)
        {
            int counter = 0;
            while(x_population.Count < count_of_individuals)
            {
                x_population.Add(values[0, counter]);
                y_population.Add(values[1, counter]);
                counter++;
            }
        }
        // Отдать x лучших значений
        public double[,] Get_the_best_values(int count_of_values)
        {
            double[,] values = new double[2, count_of_values];
            for(int i=0; i<count_of_values; i++)
            {
                int i_min_value = -1;
                double min_value = double.MaxValue;
                for (int j=0; j<f.Count; j++)
                {
                    if(f[j]<min_value)
                    {
                        min_value = f[j];
                        i_min_value = j;
                    }
                }
                values[0, i] = x_population[i_min_value];
                values[1, i] = y_population[i_min_value];
                x_population.RemoveAt(i_min_value);
                y_population.RemoveAt(i_min_value);
                f.RemoveAt(i_min_value);
            }
            return values;
        }
        // Вывод на экран наилучшего значения из всех эпох
        public void Print_best_value()
        {
            Console.WriteLine("Min value = {0}", min_value);
            Console.WriteLine("X min value = {0}", min_value_x);
            Console.WriteLine("Y min value = {0}", min_value_y);
        }


        // Функции для оптимизации
        static double Rastrigin_function(double[] mas)
        {
            int a = 10;
            double result = a * mas.Length;
            for (int i = 0; i < mas.Length; i++)
            {
                result = result + (mas[i] * mas[i] - a * Math.Cos(2 * Math.PI * mas[i]));
            }
            return result;
        }
        static double Ekli_function(double x, double y)
        {
            double result = 0;
            result = -20 * Math.Pow(Math.E, -0.2 * Math.Sqrt(0.5 * (x * x + y * y))) - Math.Pow(Math.E, 0.5 * (Math.Cos(2 * Math.PI * x) + Math.Cos(2 * Math.PI * y))) + Math.E + 20;
            return result;
        }
        static double Sphere_function(double[] mas)
        {
            double result = 0;
            for (int i = 0; i < mas.Length; i++)
            {
                result = result + mas[i] * mas[i];
            }
            return result;
        }
        static double Rozenbroke_function(double[] mas)
        {
            double result = 0;
            for (int i = 0; i < mas.Length - 1; i++)
            {
                result = result + (100 * Math.Pow((mas[i + 1] - mas[i] * mas[i]), 2) + Math.Pow(mas[i] - 1, 2));
            }
            return result;
        }
        static double Bill_function(double x, double y)
        {
            double result = 0;
            result = Math.Pow(1.5 - x + x * y, 2) + Math.Pow(2.25 - x + x * y * y, 2) + Math.Pow(2.625 - x + x * Math.Pow(y, 3), 2);
            return result;
        }
        static double Goldman_Price_function(double x, double y)
        {
            double result = 0;
            result = (1 + Math.Pow(x + y + 1, 2) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y)) * (30 + Math.Pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y));
            return result;
        }
        static double Byte_function(double x, double y)
        {
            double result = 0;
            result = Math.Pow(x + 2 * y - 7, 2) + Math.Pow(2 * x + y - 5, 2);
            return result;
        }
        static double Bukin_function(double x, double y)
        {
            double result = 0;
            result = 100 * Math.Sqrt(Math.Abs(y - 0.01 * x * x)) + 0.01 * Math.Abs(x + 10);
            return result;
        }
        static double Matias_function(double x, double y)
        {
            double result = 0;
            result = 0.26 * (x * x + y * y) - 0.48 * x * y;
            return result;
        }
        static double Levi_function(double x, double y)
        {
            double result = 0;
            result = Math.Pow(Math.Sin(3 * Math.PI * x), 2) + Math.Pow(x - 1, 2) * (1 + Math.Pow(Math.Sin(3 * Math.PI * y), 2)) + Math.Pow(y - 1, 2) * (1 + Math.Pow(Math.Sin(2 * Math.PI * y), 2));
            return result;
        }
        static double Hemmilblay_function(double x, double y)
        {
            double result = 0;
            result = Math.Pow(x * x + y - 11, 2) + Math.Pow(x + y * y - 7, 2);
            return result;
        }
        static double Camel_function(double x, double y)
        {
            double result = 0;
            result = 2 * x * x - 1.05 * Math.Pow(x, 4) + Math.Pow(x, 6) / 6 + x * y + y * y;
            return result;
        }
        static double Izome_function(double x, double y)
        {
            double result = 0;
            result = -Math.Cos(x) * Math.Cos(y) * Math.Pow(Math.E, -(Math.Pow(x - Math.PI, 2) + Math.Pow(y - Math.PI, 2)));
            return result;
        }
        static double Cross_in_tray_function(double x, double y)
        {
            double result = 0;
            result = -0.0001 * Math.Pow(Math.Abs(Math.Sin(x) * Math.Sin(y) * Math.Pow(Math.E, Math.Abs(100 - Math.Sqrt(x * x + y * y) / Math.PI))) + 1, 0.1);
            return result;
        }
        static double Egnholder_function(double x, double y)
        {
            double result = 0;
            result = -(y + 47) * Math.Sin(Math.Sqrt(Math.Abs(x / 2 + (y + 47)))) - x * Math.Sin(Math.Sqrt(Math.Abs(x - (y + 47))));
            return result;
        }
        static double table_Holder_function(double x, double y)
        {
            double result = 0;
            result = -Math.Abs(Math.Sin(x) * Math.Cos(y) * Math.Pow(Math.E, Math.Abs(1 - Math.Sqrt(x * x + y * y) / Math.PI)));
            return result;
        }
        static double MacCormic_function(double x, double y)
        {
            double result = 0;
            result = Math.Sin(x + y) + Math.Pow(x - y, 2) - 1.5 * x + 2.5 * y + 1;
            return result;
        }
        static double Shaffer_N2_function(double x, double y)
        {
            double result = 0;
            result = 0.5 + (Math.Pow(Math.Sin(x * x - y * y), 2) - 0.5) / Math.Pow(1 + 0.001 * (x * x + y * y), 2);
            return result;
        }
        static double Shaffer_N4_function(double x, double y)
        {
            double result = 0;
            result = 0.5 + (Math.Pow(Math.Cos(Math.Sin(Math.Abs(x * x - y * y))), 2) - 0.5) / Math.Pow(1 + 0.001 * (x * x + y * y), 2);
            return result;
        }
        static double Stibinskiy_Tanga_function(double[] mas)
        {
            double result = 0;
            for (int i = 0; i < mas.Length; i++)
            {
                result = result + (Math.Pow(mas[i], 4) - 16 * Math.Pow(mas[i], 2) + 5 * mas[i]);
            }
            result = result / 2;
            return result;
        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            int count_of_individuals = 250;

            //// For Rastrigin
            //double min_x = -5.12;
            //double max_x = 5.12;
            //double min_y = -5.12;
            //double max_y = 5.12;

            //// For Ecli_function
            //double min_x = -5;
            //double max_x = 5;
            //double min_y = -5;
            //double max_y = 5;

            //// For Sphere
            //double min_x = -10;
            //double max_x = 10;
            //double min_y = -10;
            //double max_y = 10;

            //// For Rozenbroke
            //double min_x = -100;
            //double max_x = 100;
            //double min_y = -100;
            //double max_y = 100;

            //// For Bill function
            //double min_x = -4.5;
            //double max_x = 4.5;
            //double min_y = -4.5;
            //double max_y = 4.5;

            //// For Goldman_Price
            //double min_x = -2;
            //double max_x = 2;
            //double min_y = -2;
            //double max_y = 2;

            //// For Buta function
            //double min_x = -10;
            //double max_x = 10;
            //double min_y = -10;
            //double max_y = 10;

            //// For Bukin
            //double min_x = -15;
            //double max_x = -5;
            //double min_y = -3;
            //double max_y = 3;

            // For Levi
            double min_x = -10;
            double max_x = 10;
            double min_y = -10;
            double max_y = 10;

            double mutation_rate = 0.1; // Вероятность мутации для каждой особи
            double death_rate = 0.05; // Вероятность смертности (умирают только худшие особи)
            int count_of_eras = 100; // Количество эпох работы генетического алгоритма
            int period_change_mutation_values = 10; // Период для изменения (уменьшения) пределов мутации
            double mean_change_mutation_values = 0.99; // Параметр изменения пределов мутации (0.995)

            //Genetic_algorithm my_algorithm = new Genetic_algorithm(count_of_individuals, count_of_eras, mutation_rate, death_rate, period_change_mutation_values, mean_change_mutation_values);

            //my_algorithm.Set_boundary_values(min_x, max_x, min_y, max_y); // Установка начальных значений

            //my_algorithm.Learning_function();

            int count_of_islands = 4;
            int count_of_change_era = 100; // Через какое количество эпох происходит обмен между островами
            int count_of_best_values = 25; // Количество лучших особей, мигрирующих с острова
            Genetic_algorithm[] my_algoritm_2 = new Genetic_algorithm[count_of_islands];
            for(int i=0; i<count_of_islands; i++)
            {
                my_algoritm_2[i] = new Genetic_algorithm(count_of_individuals, count_of_change_era, mutation_rate, death_rate, period_change_mutation_values, mean_change_mutation_values);
                my_algoritm_2[i].Set_boundary_values(min_x, max_x, min_y, max_y);
            }

            double[,,] best_values = new double[count_of_islands, 2, count_of_best_values];
            double[,] helper_mas;
            double[,] helper_mas_2 = new double[2, count_of_best_values];
            for(int i=0; i<10; i++)
            {
                Console.WriteLine("Learnig step №{0} start");
                for (int j=0; j<count_of_islands; j++)
                {
                    Console.WriteLine("Learning island №{0} start", j + 1);
                    my_algoritm_2[j].Learning_function();
                    Console.WriteLine("Learning island №{0} is end", j + 1);
                }
                Console.WriteLine("Learnig step №{0} is end. Removing start.", i + 1);
                // Перемещение особей
                for(int j=0; j<count_of_islands; j++)
                {
                    helper_mas = my_algoritm_2[j].Get_the_best_values(count_of_best_values);
                    for(int ii=0; ii<2; ii++)
                    {
                        for(int jj=0; jj<count_of_best_values; jj++)
                        {
                            best_values[j, ii, jj] = helper_mas[ii, jj];
                        }
                    }
                }
                for(int j=0; j<count_of_islands - 1; j++)
                {
                    for (int ii = 0; ii < 2; ii++)
                    {
                        for (int jj = 0; jj < count_of_best_values; jj++)
                        {
                            helper_mas_2[ii, jj] = best_values[j, ii, jj];
                        }
                    }
                    my_algoritm_2[j + 1].Set_the_best_values(helper_mas_2);
                    for (int ii = 0; ii < 2; ii++)
                    {
                        for (int jj = 0; jj < count_of_best_values; jj++)
                        {
                            helper_mas_2[ii, jj] = best_values[count_of_islands - 1, ii, jj];
                        }
                    }
                    my_algoritm_2[0].Set_the_best_values(helper_mas_2);
                }
                Console.WriteLine("Removing end.");
            }
            for(int i=0; i<count_of_islands; i++)
            {
                Console.WriteLine("Island №{0}", i + 1);
                my_algoritm_2[i].Print_best_value();
                Console.WriteLine();
            }
        }
    }
}
