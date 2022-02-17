using System;
using System.Collections.Generic;
using System.Linq;

namespace Genetic_algoritm
{
    class For_equals
    {
        public string s;
        private int count_of_signs; // Количество операторов в строке
        private string[] operator_; // Хранится символьный тип оператора, например '*'
        private string[,] variables_using; // Хранятся переменные или значения, к которым применяется данный оператор (для вычислений)
        private string[,] variables_copy; // Хранятся переменные или значения, к которым применяется данный оператор (не для вычислений)
        public double[] result; // Хранится результат выполнения операции

        int count_of_parameters = 0; // Количество параметров в системе уравнений (в одном уравнении)
        string[] parameters; // Хранятся имена переменных
        double[] value; // Хранятся значения переменных

        private List<int> coordinate_left_brace = new List<int>();
        private List<int> coordinate_right_brace = new List<int>();

        // Cтандартный конструктор
        public For_equals(string equation_str, string parameters_str)
        {
            s = equation_str;
            count_of_signs = Counting_operations();
            operator_ = new string[count_of_signs];
            variables_using = new string[count_of_signs, 2];
            variables_copy = new string[count_of_signs, 2];
            result = new double[count_of_signs];

            for (int i = 0; i < count_of_signs; i++)
            {
                operator_[i] = "";
                variables_using[i, 0] = " ";
                variables_using[i, 1] = " ";
                variables_copy[i, 0] = " ";
                variables_copy[i, 1] = " ";
                result[i] = 0;
            }

            parameters = parameters_str.Split(new char[] { ' ' });
            count_of_parameters = parameters.Length;
            value = new double[count_of_parameters];

            for (int i = 0; i < count_of_parameters; i++)
            {
                value[i] = 0;
            }
        }
        // Добавление пробелов до и перед знаками
        public string Add_space()
        {
            s = s.Insert(0, " ");
            for (int i = 1; i < s.Length; i++)
            {
                if ((s[i] == '-') || (s[i] == '+') || (s[i] == '^') || (s[i] == '*') || (s[i] == '/'))
                {
                    s = s.Insert(i, " ");
                    s = s.Insert(i + 2, " ");
                    i = i + 2;
                }
                if (s[i] == '(')
                {
                    s = s.Insert(i + 1, " ");
                    i++;
                }
                if (s[i] == ')')
                {
                    s = s.Insert(i, " ");
                    i++;
                }
                if ((s[i] == 'n') && (s[i - 1] == 'i') && (s[i - 2] == 's') && (s[i + 1] != ' '))
                {
                    s = s.Insert(i + 1, " ");
                }
                if ((s[i] == 's') && (s[i - 1] == 'o') && (s[i - 2] == 'c') && (s[i + 1] != ' '))
                {
                    s = s.Insert(i + 1, " ");
                }
                if ((s[i] == 'g') && (s[i - 1] == 't') && (s[i + 1] != ' '))
                {
                    s = s.Insert(i + 1, " ");
                }
                if ((s[i] == 'g') && (s[i - 1] == 't') && (s[i - 2] == 'c') && (s[i + 1] != ' '))
                {
                    s = s.Insert(i + 1, " ");
                }
                if ((s[i] == 'g') && (s[i - 1] == 'l'))
                {
                    s = s.Insert(i + 1, " ");
                }
                if ((s[i] == 'n') && (s[i - 1] == 'l'))
                {
                    s = s.Insert(i + 1, " ");
                }
            }
            s = s.Insert(s.Length, " ");
            return s;
        }
        // Подсчёт и упорядочивание скобочек
        public void Work_with_brace()
        {
            int size_left_brace = 0;
            int size_right_brace = 0;
            Counting_left_and_right_braces(ref size_left_brace, ref size_right_brace);

            //Print_mas(coordinate_left_brace, "Координаты левых скобочек");
            //Print_mas(coordinate_right_brace, "Координаты правых скобочек");

            // Теперь надо отсортировать массивы так, чтобы операции, ограниченные скобками, выполнялись в заданном порядке
            int max;
            int i_max;
            int helper;
            for (int i = 0; i < coordinate_left_brace.Count - 1; i++)
            {
                max = coordinate_left_brace[i];
                i_max = i;
                for (int j = i; j < coordinate_left_brace.Count; j++)
                {
                    if ((coordinate_left_brace[j] > max) && (coordinate_left_brace[j] < coordinate_right_brace[i]))
                    {
                        max = coordinate_left_brace[j];
                        i_max = j;
                    }
                }
                helper = coordinate_left_brace[i];
                coordinate_left_brace[i] = coordinate_left_brace[i_max];
                coordinate_left_brace[i_max] = helper;
            }
            //Print_mas(coordinate_left_brace, "Правильные координаты левых скобочек");
            //Print_mas(coordinate_right_brace, "Правильные координаты правых скобочек");
        }
        // Заполнение массива с операторами и переменными
        public void Filling_mass()
        {
            int begin_coordinate = 0;
            int end_coordinate = 0;
            int value = 0;

            //Console.WriteLine("Длина уравнения равна {0}", s.Length);
            //Console.WriteLine("s = {0}", s);

            // Сначала проходимся по всем знакам в скобочках
            for (int i = 0; i < coordinate_left_brace.Count; i++)
            {
                // Сначала в скобочках ищем знак '^' и обрабатываем их
                for (int j = coordinate_left_brace[i]; j < coordinate_right_brace[i]; j++)
                {
                    if (s[j] == '^')
                    {
                        j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "^");
                    }
                    else if ((s[j] == 'n') && (s[j - 1] == 'i') && (s[j - 2] == 's'))
                    {
                        if ((s[i - 3] == 'c') && (s[i - 4] == 'r') && (s[i - 5] == 'a'))
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 6, "arcsin");
                        }
                        else
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 3, "sin");
                        }
                    }
                    else if ((s[j] == 's') && (s[j - 1] == 'o') && (s[j - 2] == 'c'))
                    {
                        if ((s[i - 3] == 'c') && (s[i - 4] == 'r') && (s[i - 5] == 'a'))
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 6, "arccos");
                        }
                        else
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 3, "cos");
                        }
                    }
                    else if ((s[i] == 'g') && (s[i - 1] == 't'))
                    {
                        if ((s[i - 2] == 'c') && (s[i - 3] == 'c') && (s[i - 4] == 'r') && (s[i - 5] == 'a'))
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 6, "arcctg");
                        }
                        else if ((s[i - 2] == 'c') && (s[i - 3] == 'r') && (s[i - 4] == 'a'))
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 5, "arctg");
                        }
                        else if (s[i - 2] == 'c')
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 3, "ctg");
                        }
                        else
                        {
                            j = j - If_we_find_sign(begin_coordinate, ref value, j, 2, "tg");
                        }
                    }
                    else if ((s[i] == 'g') && (s[i - 1] == 'l'))
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 2, "lg");
                    }
                    else if ((s[i] == 'n') && (s[i - 1] == 'l'))
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 2, "ln");
                    }
                    if (j < coordinate_left_brace[i])
                    {
                        j = coordinate_left_brace[i];
                    }
                }
                // Затем в этих же скобочках ищем знаки '*' или '/' и обрабатываем их
                for (int j = coordinate_left_brace[i]; j < coordinate_right_brace[i]; j++)
                {
                    if (s[j] == '*')
                    {
                        j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "*");
                    }
                    else if (s[j] == '/')
                    {
                        j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "/");
                    }
                    if (j < coordinate_left_brace[i])
                    {
                        j = coordinate_left_brace[i];
                    }
                }
                // После чего ищем знаки '+' или '-' и обрабатываем их
                for (int j = coordinate_left_brace[i]; j < coordinate_right_brace[i]; j++)
                {
                    if (s[j] == '+')
                    {
                        j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "+");
                    }
                    else if (s[j] == '-')
                    {
                        j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "-");
                    }
                    if (j < coordinate_left_brace[i])
                    {
                        j = coordinate_left_brace[i];
                    }
                }
                // Снятие скобочек
                s = s.Remove(coordinate_left_brace[i], 2);
                s = s.Remove(coordinate_right_brace[i] - 3, 2);
                Do_shift_brace(4, coordinate_right_brace[i] - 3);
                //Print_mas(coordinate_left_brace, "Новые координаты левых скобочек");
                //Print_mas(coordinate_right_brace, "Новые координаты правых скобочек");
                //Console.WriteLine("s = {0}", s);
            }
            //Console.WriteLine("s.Length = {0}", s.Length);

            // Теперь проходимя по всем знакам без скобочек
            // Сначала в скобочках ищем знак '^' и обрабатываем их
            for (int j = 0; j < s.Length; j++)
            {
                if (s[j] == '^')
                {
                    j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "^");
                }
                else if ((s[j] == 'n') && (s[j - 1] == 'i') && (s[j - 2] == 's'))
                {
                    if ((s[j - 3] == 'c') && (s[j - 4] == 'r') && (s[j - 5] == 'a'))
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 6, "arcsin");
                    }
                    else
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 3, "sin");
                    }
                }
                else if ((s[j] == 's') && (s[j - 1] == 'o') && (s[j - 2] == 'c'))
                {
                    if ((s[j - 3] == 'c') && (s[j - 4] == 'r') && (s[j - 5] == 'a'))
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 6, "arccos");
                    }
                    else
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 3, "cos");
                    }
                }
                else if ((s[j] == 'g') && (s[j - 1] == 't'))
                {
                    if ((s[j - 2] == 'c') && (s[j - 3] == 'c') && (s[j - 4] == 'r') && (s[j - 5] == 'a'))
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 6, "arcctg");
                    }
                    else if ((s[j - 2] == 'c') && (s[j - 3] == 'r') && (s[j - 4] == 'a'))
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 5, "arctg");
                    }
                    else if (s[j - 2] == 'c')
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 3, "ctg");
                    }
                    else
                    {
                        j = j - If_we_find_sign(begin_coordinate, ref value, j, 2, "tg");
                    }
                }
                else if ((s[j] == 'g') && (s[j - 1] == 'l'))
                {
                    j = j - If_we_find_sign(begin_coordinate, ref value, j, 2, "lg");
                }
                else if ((s[j] == 'n') && (s[j - 1] == 'l'))
                {
                    j = j - If_we_find_sign(begin_coordinate, ref value, j, 2, "ln");
                }
                if (j < 0)
                {
                    j = 0;
                }
            }
            // Затем в этих же скобочках ищем знаки '*' или '/' и обрабатываем их
            for (int j = 0; j < s.Length; j++)
            {
                if (s[j] == '*')
                {
                    j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "*");
                }
                else if (s[j] == '/')
                {
                    j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "/");
                }
                if (j < 0)
                {
                    j = 0;
                }
            }
            // После чего ищем знаки '+' или '-' и обрабатываем их
            for (int j = 0; j < s.Length; j++)
            {
                if (s[j] == '+')
                {
                    j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "+");
                }
                else if (s[j] == '-')
                {
                    j = j - If_we_find_sign(begin_coordinate, end_coordinate, ref value, j, "-");
                }
                if (j < 0)
                {
                    j = 0;
                }
            }

            for (int i = 0; i < count_of_signs; i++)
            {
                variables_copy[i, 0] = variables_using[i, 0];
                variables_copy[i, 1] = variables_using[i, 1];
            }

            //Console.WriteLine("s = {0}", s);
        }
        // Вывод строки s на экран
        public void Print_s()
        {
            Console.WriteLine("s = {0}", s);
        }
        // Вывод массива на экран
        public void Print_mas(int[] x, string s)
        {
            Console.WriteLine(s);
            for (int i = 0; i < x.Length; i++)
            {
                Console.Write("{0} ", x[i]);
            }
            Console.WriteLine();
        }
        public void Print_mas(List<int> x, string s)
        {
            Console.WriteLine(s);
            for (int i = 0; i < x.Count; i++)
            {
                Console.Write("{0} ", x[i]);
            }
            Console.WriteLine();
        }
        public void Print_3_mas(string str)
        {
            Console.WriteLine(str);
            Console.WriteLine("operator_ variables[0] variables[1] result");
            for (int i = 0; i < operator_.Length; i++)
            {
                Console.WriteLine("{0} {1} {2} {3}", operator_[i], variables_using[i, 0], variables_using[i, 1], result[i]);
            }
        }
        // Подстановка значений переменных в массивы
        public void Substitution_of_Parameters()
        {
            for (int i = 0; i < count_of_signs; i++)
            {
                variables_using[i, 0] = variables_copy[i, 0];
                variables_using[i, 1] = variables_copy[i, 1];
            }

            for (int i = 0; i < count_of_signs; i++)
            {
                for (int j = 0; j < count_of_parameters; j++)
                {
                    if (variables_using[i, 0] == parameters[j])
                    {
                        variables_using[i, 0] = value[j].ToString();
                    }
                    if (variables_using[i, 1] == parameters[j])
                    {
                        variables_using[i, 1] = value[j].ToString();
                    }
                }
            }
        }
        // Выполнение операций
        public double Execute()
        {
            for (int i = 0; i < count_of_signs; i++)
            {
                if (variables_using[i, 0] == "e")
                {
                    variables_using[i, 0] = "2,718282";
                }
                if (variables_using[i, 0] == "pi")
                {
                    variables_using[i, 0] = "3,141593";
                }
                if (variables_using[i, 1] == "e")
                {
                    variables_using[i, 1] = "2,718282";
                }
                if (variables_using[i, 1] == "pi")
                {
                    variables_using[i, 1] = "3,141593";
                }
            }

            int helper_1 = 0;
            int helper_2 = 0;
            for (int i = 0; i < count_of_signs; i++)
            {
                if (variables_using[i, 0][0] == '[')
                {
                    helper_1 = int.Parse(variables_using[i, 0].Substring(1, variables_using[i, 0].IndexOf(']') - 1));
                    variables_using[i, 0] = result[helper_1].ToString();
                }
                if (variables_using[i, 1][0] == '[')
                {
                    helper_2 = int.Parse(variables_using[i, 1].Substring(1, variables_using[i, 1].IndexOf(']') - 1));
                    variables_using[i, 1] = result[helper_2].ToString();
                }

                if (operator_[i] == "^")
                {
                    result[i] = Math.Pow(double.Parse(variables_using[i, 0]), double.Parse(variables_using[i, 1]));
                }
                else if (operator_[i] == "*")
                {
                    result[i] = double.Parse(variables_using[i, 0]) * double.Parse(variables_using[i, 1]);
                }
                else if (operator_[i] == "/")
                {
                    result[i] = double.Parse(variables_using[i, 0]) / double.Parse(variables_using[i, 1]);
                }
                else if (operator_[i] == "+")
                {
                    result[i] = double.Parse(variables_using[i, 0]) + double.Parse(variables_using[i, 1]);
                }
                else if (operator_[i] == "-")
                {
                    result[i] = double.Parse(variables_using[i, 0]) - double.Parse(variables_using[i, 1]);
                }
                else if (operator_[i] == "sin")
                {
                    result[i] = Math.Sin(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "cos")
                {
                    result[i] = Math.Cos(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "tg")
                {
                    result[i] = Math.Tan(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "ctg")
                {
                    result[i] = 1 / (Math.Tan(double.Parse(variables_using[i, 0])));
                }
                else if (operator_[i] == "arcsin")
                {
                    result[i] = Math.Asin(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "arccos")
                {
                    result[i] = Math.Acos(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "arctg")
                {
                    result[i] = Math.Atan(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "arcctg")
                {
                    result[i] = Math.PI / 2 - Math.Atan(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "ln")
                {
                    result[i] = Math.Log(double.Parse(variables_using[i, 0]));
                }
                else if (operator_[i] == "lg")
                {
                    result[i] = Math.Log10(double.Parse(variables_using[i, 0]));
                }
            }
            //Console.WriteLine("Ответ: {0}", result[count_of_signs - 1]);
            return result[count_of_signs - 1];
        }
        // Обработка поступившего уравнения
        public void Decide_first_time()
        {
            Add_space(); // Добавление пробелов до и перед знаками
            Print_s(); // Вывод строки s на экран

            Work_with_brace(); // Подсчёт и упорядочивание скобочек
            Filling_mass(); // Заполнение массива с операторами и переменными
            //Print_3_mas("Массивы после второго этапа преобразований"); // Вывод строки s на экран
        }
        // Вычисление значения уравнения(ий) с заданными параметрами
        public double Decide(double[] mas)
        {
            Change_mas_of_string(mas);

            Substitution_of_Parameters(); // Подстановка значений переменных в массивы
            //Print_3_mas("Массивы после третьего этапа преобразований"); // Вывод строки s на экран

            return Execute();
        }
        public double Decide(string[] mas)
        {
            Change_mas_of_string(mas);

            Substitution_of_Parameters(); // Подстановка значений переменных в массивы
            //Print_3_mas("Массивы после третьего этапа преобразований"); // Вывод строки s на экран

            return Execute();
        }
        // Замена значений переменных
        public void Change_mas_of_string(double[] mas)
        {
            for (int i = 0; i < count_of_parameters; i++)
            {
                value[i] = mas[i];
            }
            //Console.WriteLine("mas_of_string[0] = {0}", value[0]);
        }
        public void Change_mas_of_string(string[] mas)
        {
            for (int i = 0; i < count_of_parameters; i++)
            {
                value[i] = double.Parse(mas[i]);
            }
            //Console.WriteLine("mas_of_string[0] = {0}", value[0]);
        }
        // Удаление пробелов из строки
        private string Delete_space()
        {
            //Console.WriteLine("До удаления пробелов: {0}", s);
            string s_copy = s;
            s_copy = s_copy.Replace(" ", "");
            //Console.WriteLine("После удаления пробелов: {0}", s_copy);
            return s_copy;
        }
        // Поиск последнего пробела в строке
        private int Search_last_space(int first_number)
        {
            for (int i = first_number; i < s.Length; i++)
            {
                if (s[i] == ' ')
                {
                    return i;
                }
            }
            return s.Length;
        }
        // Поиск первого пробела в строке
        private int Search_first_space(int last_number)
        {
            int number = -1;
            for (int i = 0; i < last_number; i++)
            {
                if (s[i] == ' ')
                {
                    number = i;
                }
            }
            return number;
        }
        // Посчёт количества знаков в уравнении
        private int Counting_operations()
        {
            int lol = s.Length;
            int count_of_signs = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if ((s[i] == '-') || (s[i] == '+') || (s[i] == '^') || (s[i] == '*') || (s[i] == '/'))
                {
                    count_of_signs++;
                }
            }
            // Подсчитываем количество синусов в строке
            count_of_signs = count_of_signs + ((s.Length - s.Replace("sin", "").Length) / 3);
            count_of_signs = count_of_signs + ((s.Length - s.Replace("cos", "").Length) / 3);
            count_of_signs = count_of_signs + ((s.Length - s.Replace("tg", "").Length) / 2);
            count_of_signs = count_of_signs + ((s.Length - s.Replace("ln", "").Length) / 2);
            count_of_signs = count_of_signs + ((s.Length - s.Replace("lg", "").Length) / 2);
            return count_of_signs;
        }
        // Делаем сдвиг массива с координатами скобочек
        private void Do_shift_brace(int shift, int j)
        {
            for (int k = 0; k < coordinate_left_brace.Count; k++)
            {
                if (coordinate_left_brace[k] > j)
                {
                    coordinate_left_brace[k] = coordinate_left_brace[k] - shift;
                }
                if (coordinate_right_brace[k] > j)
                {
                    coordinate_right_brace[k] = coordinate_right_brace[k] - shift;
                }
            }
        }
        // Заполнение массивов информацией, если мы нашли знак
        private int If_we_find_sign(int begin_coordinate, ref int value, int j, int length, string sign)
        {
            begin_coordinate = Search_first_space(j + 2);

            operator_[value] = sign;
            variables_using[value, 0] = s.Substring(j + 2, j + 2 - begin_coordinate);

            s = s.Remove(j - length + 1, begin_coordinate - j + length + 1);
            string helper = value.ToString();
            value++;
            helper = "[" + helper + "]";
            s = s.Insert(j - length + 1, helper);

            // Необходимо изменить положение скобочек в строке
            int shift = (begin_coordinate - j + length - 1) - helper.Length;
            Do_shift_brace(shift, j);

            return shift;
        }
        private int If_we_find_sign(int begin_coordinate, int end_coordinate, ref int value, int j, string sign)
        {
            // До и после знака идёт пробел. Нужно найти ещё соседние пробелы справа и слева.
            begin_coordinate = Search_first_space(j - 2);
            end_coordinate = Search_last_space(j + 2);

            operator_[value] = sign;

            variables_using[value, 0] = s.Substring(begin_coordinate + 1, j - 2 - begin_coordinate);
            variables_using[value, 1] = s.Substring(j + 2, end_coordinate - j - 2);

            s = s.Remove(begin_coordinate + 1, end_coordinate - begin_coordinate - 1);
            string helper = value.ToString();
            value++;
            helper = "[" + helper + "]";
            s = s.Insert(begin_coordinate + 1, helper);

            // Необходимо изменить положение скобочек в строке
            int shift = (end_coordinate - begin_coordinate - 1) - helper.Length;
            // Проходимся по всем элементам массива, в котором хранятся координаты скобочек и, если элемент массива
            // больше текущего, то делаем сдвиг
            Do_shift_brace(shift, j);

            return shift;
        }
        // Считаем количество правых и левых скобочек
        private void Counting_left_and_right_braces(ref int size_left_brace, ref int size_right_brace)
        {
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    size_left_brace++;
                    coordinate_left_brace.Add(i);
                }
                if (s[i] == ')')
                {
                    size_right_brace++;
                    coordinate_right_brace.Add(i);
                }
            }
        }
    }
    class Genetic_algorithm
    {
        // Инициализируем переменные
        // Рассмотрим функцию Экли
        int count_of_individuals = 1000; // Количество особей в популяции
        int count_of_parameters = 0; // Количество переменных в системе
        double[] min_parameters;
        double[] max_parameters;
        //double min_x = -5; // Минимальное значение х
        //double max_x = 1000; // Максимальное значение x
        //double min_y = -1000; // Минимальное значение y
        //double max_y = 5; // Максимальное значение y
        double mutation_rate = 0.05; // Вероятность мутации для каждой особи
        double[] mutation_parameters;
        //double mutation_x = 0; // Начальное значение мутации по x
        //double mutation_y = 0; // Начальное значение мутации по y
        int period_change_mutation_values = 10; // Период для изменения (уменьшения) пределов мутации
        double mean_change_mutation_values = 0.5; // Параметр изменения пределов мутации
        double death_rate = 0.05; // Вероятность смертности (умирают только худшие особи)
        int count_of_eras = 1000; // Количество эпох работы генетического алгоритма

        //List<double> x_population = new List<double>(); // Значения параметров х
        //List<double> y_population = new List<double>(); // Значения параметров у
        List<double>[] population; // Массив списков с переменными популяции
        List<double> f = new List<double>(); // Значения функции приспособленности
        List<double> p = new List<double>(); // Значения вероятностей выбора i значения для дальнейшего отбора (селекции) (плотность распределения)
        List<double> P = new List<double>(); // Значения первообразных от вероятностей (значения функции распределения)

        double min_value = double.MaxValue; // Лучшее (минимальное значение из всех эпох обучения)
        int era_min_value = -1; // Эпоха, в которое было достигнуто минимальное значение
        double[] overall_min_parameters;
        //double min_value_x = 0; // Значение переменной x для наименьшего значения функции
        //double min_value_y = 0; // Значение переменной y для наименьшего значения функции
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
        //public void Set_boundary_values(double min_x_, double max_x_, double min_y_, double max_y_)
        //{
        //    min_x = min_x_;
        //    max_x = max_x_;
        //    min_y = min_y_;
        //    max_y = max_y_;

        //    mutation_x = (max_x - min_x) / count_of_individuals;
        //    mutation_y = (max_y - min_y) / count_of_individuals;
        //}
        public void Set_boundary_values(double[] min_parameters_, double[] max_parameters_)
        {
            if (min_parameters_.Length != max_parameters_.Length)
            {
                Console.WriteLine("Some problems with lelgth");
            }
            count_of_parameters = min_parameters_.Length;
            min_parameters = min_parameters_;
            max_parameters = max_parameters_;

            mutation_parameters = new double[min_parameters.Length];
            overall_min_parameters = new double[count_of_parameters];
            population = new List<double>[count_of_parameters];

            for (int i = 0; i < count_of_parameters; i++)
            {
                population[i] = new List<double>() { };
            }

            for (int i = 0; i < mutation_parameters.Length; i++)
            {
                mutation_parameters[i] = (max_parameters[i] - min_parameters[i]) / count_of_individuals;
            }
        }
        // Получение начальной популяции случайным образом в заданных пределах
        public void Get_random_start_population()
        {
            var rnd = new Random();
            //for (int i = 0; i < count_of_individuals; i++)
            //{
            //    x_population.Add(rnd.NextDouble() * (max_x - min_x) + min_x);
            //    y_population.Add(rnd.NextDouble() * (max_y - min_y) + min_y);
            //}

            for (int i = 0; i < count_of_parameters; i++)
            {
                for (int j = 0; j < count_of_individuals; j++)
                {
                    population[i].Add(rnd.NextDouble() * (max_parameters[i] - min_parameters[i]) + min_parameters[i]);
                }
            }
        }
        // Получение начальной популяции равномерно с заданным шагом
        public void Get_periodic_start_population()
        {
            double[] parameters_step = new double[count_of_parameters];
            population = new List<double>[count_of_parameters];

            for (int i = 0; i < count_of_parameters; i++)
            {
                parameters_step[i] = (max_parameters[i] - min_parameters[i]) / count_of_individuals;
            }

            for (int i = 0; i < count_of_parameters; i++)
            {
                for (double j = min_parameters[i]; j <= max_parameters[i]; j = j + parameters_step[i])
                {
                    Console.WriteLine(population[0]);
                    population[i].Add(j);
                }
            }


            //double step_x = (max_x - min_x) / count_of_individuals;
            //double step_y = (max_y - min_y) / count_of_individuals;

            //int counter = 0;
            //for(double x=min_x; x<=max_x; x=x+step_x)
            //{
            //    x_population.Add(x);
            //    counter++;
            //}
            //counter = 0;
            //for(double y=min_y; y<=max_y; y=y+step_y)
            //{
            //    y_population.Add(y);
            //    counter++;
            //}
        }
        // Вывод популяции на экран
        //public void Print_value_population()
        //{
        //    for (int i = 0; i < f.Count; i++)
        //    {
        //        Console.WriteLine("{0}, {1}, {2}", x_population[i], y_population[i], f[i]);
        //    }
        //}
        public void Print_value_population_2()
        {
            for (int i = 0; i < f.Count; i++)
            {
                for (int j = 0; j < count_of_parameters; j++)
                {
                    Console.Write("{0} ", population[j][i]);
                }
                Console.WriteLine(f[i]);
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
            for (int i = 0; i < mas_min_value.Count; i++)
            {
                Console.WriteLine("Era = {0}. Value = {1}", i + 1, mas_min_value[i]);
            }
            Console.WriteLine();
            Console.WriteLine("Min value from all eras = {0}", min_value);
            for (int i = 0; i < count_of_parameters; i++)
            {
                Console.WriteLine("Min parameter[{0}] = {1}", i + 1, overall_min_parameters[i]);
            }
            //Console.WriteLine("Min x = {0}", min_value_x);
            //Console.WriteLine("Min y = {0}", min_value_y);
            Console.WriteLine("Era min value = {0}", era_min_value);
        }
        // Вычисление значений популяции
        public void Calculation_population_values()
        {
            f.Clear();
            double[] helper_mas = new double[2];
            for (int i = 0; i < count_of_individuals; i++)
            {
                helper_mas[0] = population[0][i];
                helper_mas[1] = population[1][i];

                //f.Add(Ekli_function(x_population[i], y_population[i]));
                f.Add(Sphere_function(helper_mas));
                //f.Add(Bill_function(x_population[i], y_population[i]));
            }

            //for (int i = 0; i < count_of_individuals; i++)
            //{
            //    //f.Add(Ekli_function(x_population[i], y_population[i]));
            //    f.Add(Bill_function(population[0][i], population[1][i]));
            //}

            //Console.WriteLine("len(f) = {0}", f.Count);
        }
        // Нахождение минимального значения из всех особей
        public void Get_population_min_value(int value_era)
        {
            double min_value_in_era = double.MaxValue;
            double min_value_x_in_era = 0;
            double min_value_y_in_era = 0;

            double[] min_parameters_in_era = new double[count_of_parameters];
            for (int i = 0; i < count_of_parameters; i++)
            {
                min_parameters_in_era[i] = 0;
            }


            for (int i = 0; i < f.Count; i++)
            {
                if (f[i] < min_value_in_era)
                {
                    //min_value_in_era = f[i];
                    //min_value_x_in_era = x_population[i];
                    //min_value_y_in_era = y_population[i];

                    for (int j = 0; j < count_of_parameters; j++)
                    {
                        min_parameters_in_era[j] = population[j][i];
                    }
                }
            }
            mas_min_value.Add(min_value_in_era);
            if (min_value_in_era < min_value)
            {
                min_value = min_value_in_era;

                //min_value_x = min_value_x_in_era;
                //min_value_y = min_value_y_in_era;

                for (int i = 0; i < count_of_parameters; i++)
                {
                    overall_min_parameters[i] = min_parameters_in_era[i];
                }

                era_min_value = value_era;
            }
        }
        // Смерть особей популяции
        public void Death()
        {
            int i_max = -1; // Индекс с максимальным значением элемента
            double max = double.MinValue;
            for (int i = 0; i < (int)count_of_individuals * death_rate; i++)
            {
                i_max = -1;
                max = double.MinValue;
                for (int j = 0; j < f.Count; j++)
                {
                    if (f[j] > max)
                    {
                        max = f[j];
                        i_max = j;
                    }
                }
                //Console.WriteLine("i_max = {0}", i_max);
                //Console.WriteLine("f.Count = {0}", f.Count);
                //Console.WriteLine("x_population.Count = {0}", x_population.Count);
                //Console.WriteLine("y_population.Count = {0}", y_population.Count);

                //x_population.RemoveAt(i_max);
                //y_population.RemoveAt(i_max);

                for (int j = 0; j < count_of_parameters; j++)
                {
                    population[j].RemoveAt(i_max);
                }

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
            for (int i = 0; i < f.Count; i++)
            {
                F_x.Add(1 + (f[i] - f.Average()) / (2 * sigma));
                //Console.WriteLine("{0}", F_x[i]);
            }

            p.Clear();
            for (int i = 0; i < f.Count; i++)
            {
                p.Add(F_x[i] / F_x.Sum());
            }

            //Console.WriteLine("P =");
            P.Clear();
            double sum = 0;
            for (int i = 0; i < f.Count; i++)
            {
                sum = sum + p[i];
                P.Add(sum);
                //Console.WriteLine(P[i]);
            }

            //Console.WriteLine("P.Count = {0}", P.Count);
            //Console.WriteLine("x_population.Count = {0}", x_population.Count);
            //Console.WriteLine("y_population.Count = {0}", y_population.Count);

            var r = new Random();
            while (population[0].Count < count_of_individuals)
            {
                //// Выбор родителя №1
                //int number_parent_1 = 0;
                //int number_parent_2 = 0;
                //while ((P[number_parent_1] < r.NextDouble()) && (number_parent_1 < P.Count))
                //{
                //    number_parent_1++;
                //}

                //// Выбор родителя №2
                //while ((P[number_parent_2] < r.NextDouble()) && (number_parent_2 < P.Count))
                //{
                //    number_parent_2++;
                //}

                //// Создание новой особи
                //double x_gen_percentage = r.NextDouble();
                //double y_gen_percentage = r.NextDouble();
                //x_population.Add(x_gen_percentage * x_population[number_parent_1] + (1 - x_gen_percentage) * x_population[number_parent_2]);
                //y_population.Add(y_gen_percentage * y_population[number_parent_1] + (1 - y_gen_percentage) * y_population[number_parent_2]);

                double[] gen_percentage = new double[count_of_parameters];
                int[] number_parents = new int[count_of_parameters];
                for (int i = 0; i < count_of_parameters; i++)
                {
                    while ((P[number_parents[i]] < r.NextDouble()) && (number_parents[i] < P.Count))
                    {
                        number_parents[i]++;
                    }
                }

                for (int i = 0; i < count_of_parameters; i++)
                {
                    gen_percentage[i] = r.NextDouble();
                    population[i].Add(gen_percentage[i] * population[i][number_parents[i]] + (1 - gen_percentage[i]) * population[i][number_parents[i]]);
                }
            }

            //Console.WriteLine("x_population.Cout = {0}", x_population.Count);
            //Console.WriteLine("y_population.Cout = {0}", y_population.Count);
        }
        // Метод, организующий мутацию
        public void Mutation()
        {
            List<int> index = new List<int>(); // Значения индексов
            List<int> mutation_index_x = new List<int>(); // Значения индексов x для мутации
            List<int> mutation_index_y = new List<int>(); // Значения индексов y для мутации


            List<int>[] mutation_index = new List<int>[count_of_parameters];
            for (int i = 0; i < count_of_parameters; i++)
            {
                for (int j = 0; j < population[0].Count; j++)
                {
                    index.Add(i);
                }
                var r = new Random();
                int value_index = 0;
                for (int j = 0; j < population[i].Count * mutation_rate; i++)
                {
                    value_index = (int)(r.NextDouble() * index.Count);
                    mutation_index[i].Add(value_index);
                    index.RemoveAt(value_index);
                }
                index.Clear();

                double mutation_value = 0;
                for (int j = 0; j < mutation_index[i].Count; j++)
                {
                    mutation_value = r.NextDouble() * mutation_parameters[j] - mutation_parameters[j] / 2;
                    population[j][mutation_index[i][j]] = population[j][mutation_index[i][j]] + mutation_value;
                }
            }


            //for(int i=0; i<x_population.Count; i++)
            //{
            //    index.Add(i);
            //}

            //// Выбор значений x из популяции для дальнейшей мутации
            //var r = new Random();
            //int value_index = 0;
            //for(int i = 0; i < x_population.Count * mutation_rate; i++)
            //{
            //    value_index = (int)(r.NextDouble() * index.Count);
            //    mutation_index_x.Add(value_index);
            //    index.RemoveAt(value_index);
            //}

            //index.Clear();
            //for (int i = 0; i < x_population.Count; i++)
            //{
            //    index.Add(i);
            //}

            //for (int i = 0; i < y_population.Count * mutation_rate; i++)
            //{
            //    value_index = (int)(r.NextDouble() * index.Count);
            //    mutation_index_y.Add(value_index);
            //    index.RemoveAt(value_index);
            //}

            //// Процесс мутации
            //double mutation_value_x = 0; // Значение, на которое происходит мутация особи по x
            //double mutation_value_y = 0; // Значение, на которое происходит мутация особи по y
            //for (int i=0; i<mutation_index_x.Count; i++)
            //{
            //    mutation_value_x = r.NextDouble() * mutation_x - mutation_x / 2;
            //    mutation_value_y = r.NextDouble() * mutation_y - mutation_y / 2;
            //    x_population[mutation_index_x[i]] = x_population[mutation_index_x[i]] + mutation_value_x;
            //    y_population[mutation_index_y[i]] = y_population[mutation_index_y[i]] + mutation_value_y;
            //}
        }
        // Обучающая функция
        public void Learning_function()
        {
            //Get_random_start_population();
            Get_periodic_start_population(); // Получаем начальную популяцию

            int value_era = 0;
            while (value_era < count_of_eras)
            {
                Console.WriteLine("Era of learning №{0}", value_era + 1);

                Calculation_population_values(); // Вычисляем значения начальной популяции
                //Print_value_population();
                Get_population_min_value(value_era);

                //Print_sort_value_population();
                Death(); // Убиваем неприспособившиеся особи
                //Print_sort_value_population();
                Selection_sigma_clipping(); // Рождение новых особей с помощью метода сигмы-отсечения
                Mutation(); // Мучация части популяции

                if ((value_era + 1) % period_change_mutation_values == 0)
                {
                    //mutation_x = mutation_x * mean_change_mutation_values;
                    //mutation_y = mutation_y * mean_change_mutation_values;
                    for (int i = 0; i < count_of_parameters; i++)
                    {
                        mutation_parameters[i] = mutation_parameters[i] * mean_change_mutation_values;
                    }
                }

                value_era++;
            }
            Print_min_values();
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
            //// Код для тестирования
            //List<int> a = new List<int>();
            //a.Add(10);
            //List<int> b = new List<int>(a);
            //b.Add(100);
            //for(int i=0; i<a.Count; i++)
            //{
            //    Console.WriteLine(a[i]);
            //}


            // Основной код программы
            // Инициализируем переменные
            // Рассмотрим функцию Экли
            int count_of_individuals = 1000;

            // For Billa
            //double min_x = -4.5;
            //double max_x = 4.5;
            //double min_y = -4.5;
            //double max_y = 4.5;

            // For Sphere
            double min_x = -10;
            double max_x = 10;
            double min_y = -10;
            double max_y = 10;

            double[] min = new double[] { -10, -10 };
            double[] max = new double[] { 10, 10 };

            double mutation_rate = 0.1; // Вероятность мутации для каждой особи
            double death_rate = 0.05; // Вероятность смертности (умирают только худшие особи)
            int count_of_eras = 5000; // Количество эпох работы генетического алгоритма
            int period_change_mutation_values = 10; // Период для изменения (уменьшения) пределов мутации
            double mean_change_mutation_values = 0.99; // Параметр изменения пределов мутации (0.995)

            Genetic_algorithm my_algorithm = new Genetic_algorithm(count_of_individuals, count_of_eras, mutation_rate, death_rate, period_change_mutation_values, mean_change_mutation_values);

            //my_algorithm.Set_boundary_values(min_x, max_x, min_y, max_y); // Установка начальных значений
            my_algorithm.Set_boundary_values(min, max);

            my_algorithm.Learning_function();
        }
    }
}
