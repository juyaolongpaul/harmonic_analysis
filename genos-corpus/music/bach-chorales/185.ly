\version "2.10.33"

\header {
  title = "185 - Nun freut euch, Gottes Kinder all'"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c' {
  \partial 4 d4 
    d4 c4 f4 g4 
  a4. gis8 a4 b4 
  c4 b4 c4 a4 
  b8 d4 cis8 d4 d4 
  c4 a4 f4 d4   %5
  e4 d4 cis4 e4 
  g8 f8 e4 f4 d4 
  e4. e8 d4
  
}

alto = \relative c' {
  \partial 4 a4 
    a8 b8 c4 d4 e4 
  e4 e4 e4 g4 
  g4 g4 g4. f16 e16 
  d4 e4 fis4 g8 f8 
  e4 c4 c4 bes4~   %5
  bes8 a8 b4 a4 cis4 
  d4 a4 a4 bes4~
  bes8 d4 cis8 a4
  
}

tenor = \relative c {
  \partial 4 f4 
    f4 g4 a8 f8 c'4~ 
  c8 b16 a16 b8 d8 c4 d4 
  e4 f8 d8 c4 c8 d16 c16 
  b8 a8 g8 a8 a4 d4 
  g,4. f8 f4 f4   %5
  g4 f8 e8 e4 a4 
  g4. g8 f4 f4 
  g8 bes8 a8. g16 fis4
  
}

baixo = \relative c {
  \partial 4 d8  e8 
    f4. e8 d4 c8 b8 
  a4 e'4 a,4 g'8 f8 
  e8 c8 d8 f8 e8 c8 f8 d8 
  g8 f8 e8 a8 d,4 b4 
  c8 e8 f4 a,8 f8 bes8 a8   %5
  g4 gis4 a4 a4 
  b4 cis4 d8 c8 bes8 a8 
  g4 a4 d,
  
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