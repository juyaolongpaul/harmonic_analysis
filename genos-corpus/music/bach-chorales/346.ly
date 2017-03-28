\version "2.10.33"

\header {
  title = "346 - Meines Lebens letzte Zeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  b4 b4 c4 b4 
  b4 a4 b2 
  d4 d4 e4 e4 
  a,4 d8 c8 b4 b4 
  g4. a8 b4 a8 g8   %5
  fis4. e8 e2 
  b'4. c8 d4 b4 
  c4 b8 a8 a4 g4 
  b4 b4 e4 e4 
  a,4 a4 b4 c4   %10
  d4 d4 c4 b4 
  a4 a4 b4. c8 
  d4 b4 c4 b4 
  a4 b4 b4 a8 g8 
  fis2 e2   %15
  
}

alto = \relative c'' {
  g4 g4 fis4 g8 fis8 
  e8 dis8 e4 dis2 
  g8 a8 b8 a8 b4 e,4 
  d8 g8 fis4 g4 g4 
  e4 e4 fis4 e4   %5
  e4 dis4 b2 
  g'4 g4 a4 g4~ 
  g8 fis8 g4. fis8 d4 
  g8 a8 b4 b4 a8 g8~ 
  g8 fis16 e16 fis4 g4 g4   %10
  a8 g8 a4 g8 a4 g8 
  g8 fis16 e16 fis4 g4 g4 
  gis8 fis8 gis8 b4 a8 g4~ 
  g8 fis8 fis4 e4 e4~ 
  e4 dis4 b2   %15
  
}

tenor = \relative c' {
  e4 e4 a,4 g4 
  g4 fis8 e8 fis2 
  b8 c8 d2 c4 
  d4 d4 d4 d4 
  c4 c4 b4 b4   %5
  c4 b8. a16 g2 
  e'4 d4 d4 d4 
  c8 d8 e4 d8. c16 b4 
  d4 g8 fis8 e8 d8 e4 
  d4 d4 d4 e4   %10
  a,4 d8 fis8 e8 d8 d4 
  d4 d4 d4 e4 
  b4 e4 e4 d4 
  d4 b8 a8 g4 a8 b8 
  c4 b8 a8 gis2   %15
  
}

baixo = \relative c, {
  e4 e'4. dis8 e8 g8 
  c,2 b2 
  g4 g'4. fis8 g8 c8 
  fis,8 e8 d4 g4 g,4 
  c'8 b8 c4 dis,4 e4   %5
  a,4 b4 e,2 
  e'8 fis8 g4. fis8 g8 b8 
  e,8 d8 c4 d4 g,4 
  g'8 fis8 e8 d8 c4 cis4 
  d2 g4 fis8 e8   %10
  fis8 e8 fis8 b8 e,8 fis8 g4 
  d2 g8 fis8 e4~ 
  e8 d8 e8 gis8 a,4 b8 c8 
  d4 dis4 e8 g8 c,8 b8 
  a4 b4 e2   %15
  
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