\version "2.10.33"

\header {
  title = "154 - Der du bist drei in Einigkeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 a4 
    g4 d4 f4 g4 
  a4 a4 g4 a4 
  g4 d4 f4 g4 
  a4 a4 g4 a4 
  b4 c4 a4 g4   %5
  g4 f4 e4 f4 
  g4 d4 f4 g4 
  a4 a4 g
}

alto = \relative c' {
  \partial 4 d4~
  d8 c4 b16 cis16 d8 c8 
  b4 e4 d8 c8 b4 
  d4~
  d8 c4 b16 cis16 d8 c8 
  b4 e4 d4 d4 
  f4 f4. e4 d8   %5
  d4 e4. d8 cis4 
  d8 c8 b8 c4 b16 cis16 d8 c8 
  b4 e4 d8 c8 b4
}

tenor = \relative c {
  \partial 4 f4 
    g4 g4 a4 g4~ 
  g4 fis4 d4 f4 
  g4 g4 a4 g4~ 
  g4 fis8 c'8 b4 c4 
  d4 g,4 a16 b16 c4 b8   %5
  c8 bes8 a4 a4 a4 
  g4 g4 a4 g4~ 
  g4 fis4 d
}

baixo = \relative c {
  \partial 4 d4 
    e4 g4 d4 e8 d8 
  c4 d4 g,4 d'4 
  e4 g4 d4 e8 d8 
  c4 d4 g,4 f'8 e8 
  d4 c4 f8 fis8 g4   %5
  c,8 cis8 d4 a4 d4 
  e4 g4 d4 e8 d8 
  c4 d4 g,
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}