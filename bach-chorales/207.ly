\version "2.10.33"

\header {
  title = "207 - Des heil'gen Geistes reiche Gnad'"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \minor
}


soprano = \relative c' {
  d4 d4 d4 
  a'2 a4 
  bes2 bes4 
  a2. 
  f4 g4 a4   %5
  bes2. 
  a4 g4. f8 
  f2 g4 
  a2 g4 
  g2 a4   %10
  f2 f4 
  e2. 
  d4 e4 f4 
  g2. 
  f4 e4. d8   %15
  d2. 
  
}

alto = \relative c' {
  a4 a4 d4 
  e4 d4 a'4~ 
  a4 g8 fis8 g4~ 
  g4 f2 
  f4. e8 ees8 d8   %5
  d4 e8 f8 g4~ 
  g8 f8 f4 e4 
  c2 e4 
  f2 e8 d8 
  cis8 d8 e4 e4~   %10
  e4 d8 cis8 d4 
  d4 cis2 
  d4 a4 d4 
  e4. d8 e4~ 
  e8 d8 d4 cis4   %15
  a2. 
  
}

tenor = \relative c {
  f8 g8 a8 g8 a4~ 
  a8 g8 f8 e8 d4 
  d'2 d4 
  d2. 
  a8 d4 c8 c4   %5
  bes8 a8 g4 c4 
  c4 d4 c8. bes16 
  a2 c4 
  c2 c8 bes8 
  bes4 a8 b8 cis4   %10
  a4 a4 b4 
  e,4 a2 
  a4 a4 a4 
  bes4 a4 a4 
  a4 bes4 a8. g16   %15
  fis2. 
  
}

baixo = \relative c {
  d8 e8 f8 e8 f8 d8 
  cis4 d4 fis,4 
  g8 a8 bes4 g4 
  d'2. 
  d4 e4 fis4   %5
  g4 c,8 d8 e4 
  f4 bes,4 c4 
  f,2 c'4 
  f,8 g8 a8 bes8 c8 d8 
  e8 d8 cis4 a4   %10
  d8 e8 f4 gis,4 
  a2. 
  f'8 e8 d8 cis8 d4~ 
  d4 cis8 b8 cis8 a8 
  d4 g,4 a4   %15
  d,2. 
  
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