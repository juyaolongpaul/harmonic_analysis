\version "2.10.33"

\header {
  title = "75 - Das walt' mein Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c' {
  \partial 4 f4 
    f4 e4 f2 
  a4 c4 bes4 a4 
  g4 g4 g4 c,4 
  d8 e8 f4 f4 e4 
  f2 r4 f8 g8   %5
  a4 g4 a4 bes4 
  c2 g4 g4 
  g4 g4 g4 g4 
  a2 a4 f4 
  e4 d4 d4 cis4   %10
  d2.
  
}

alto = \relative c' {
  \partial 4 d4 
    c4 c4 c2 
  f4 g4 f8 g8 g8 f8~ 
  f8 e8 d4 c4 c4 
  bes4 c8 d8 c4 c4 
  c2 r4 c4   %5
  f4 g4 f4 f4 
  f2 e4 g8 f8 
  e4 d4 e4 e4 
  e8 d8 e8 cis8 d4 d8 c8 
  bes4 a4 bes4 a4   %10
  a2.
  
}

tenor = \relative c' {
  \partial 4 a4 
    a4 g16 a16 bes8 a2 
  c4 c4 d8 c8 c4 
  c4 g8 f4 e8 f4 
  f8 g8 a4 g8 a8 bes4 
  a2 r4 a8 bes8   %5
  c4 c4 c4 d4 
  c2 c4 d4 
  c4 b4 c4 c4 
  cis8 b8 a8 g8 f4 a4 
  g4. f8 e4. g8   %10
  fis2. 
  
}

baixo = \relative c {
  \partial 4 d4 
    a8 bes8 c4 f,2 
  f'4. e8 d8 e8 f4 
  c4 b4 c4 a4 
  bes4 a8 bes8 c4 c,4 
  f2 r4 f4   %5
  f'4. e8 f8 e8 d4 
  a8 g8 a8 bes8 c4 b4 
  c8 e8 g8 f8 e8 d8 c8 bes8 
  a8 b8 cis8 a8 d4 d4~ 
  d8 cis8 d8 f,8 g8 e8 a4   %10
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